import io
import cv2
import numpy as np
from datetime import datetime
from urllib.parse import urlparse, unquote

from google.cloud import storage
from google.cloud import firestore
from ultralytics import YOLO

from embeddings import get_embedding

# ================= CONFIG =================
BUCKET_NAME = "smart-cat-water-bowl.firebasestorage.app"
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.4

COLLECTION = "ai_training_queue"

# =========================================
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

db = firestore.Client()
yolo = YOLO(YOLO_MODEL_PATH)


# ---------- utils ----------
def firebase_url_to_gcs_path(url: str) -> str:
    parsed = urlparse(url)
    encoded = parsed.path.split("/o/")[1]
    return unquote(encoded)


def load_image_from_gcs(gcs_path: str) -> np.ndarray:
    blob = bucket.blob(gcs_path)
    data = blob.download_as_bytes()

    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_cat_crops(image: np.ndarray):
    results = yolo(image, conf=YOLO_CONF, verbose=False)
    crops = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if yolo.names[cls] != "cat":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                crops.append(crop)

    return crops


def save_embedding(owner_uid: str, cat_id: str, embedding: np.ndarray):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    path = f"embeddings/{owner_uid}/{cat_id}/emb_{ts}.npy"

    buf = io.BytesIO()
    np.save(buf, embedding)
    buf.seek(0)

    blob = bucket.blob(path)
    blob.upload_from_file(buf, content_type="application/octet-stream")

    return path


# ---------- main process ----------
def process_one_document(doc):
    data = doc.to_dict()
    owner_uid = data["ownerUid"]
    cat_id = data["catId"]
    images = data.get("images", [])

    print(f"▶ Processing cat {cat_id}")

    # mark as processing
    doc.reference.update({
        "status": "processing",
        "updatedAt": firestore.SERVER_TIMESTAMP
    })

    saved_count = 0

    for url in images:
        try:
            gcs_path = firebase_url_to_gcs_path(url)
            image = load_image_from_gcs(gcs_path)
            crops = detect_cat_crops(image)

            if not crops:
                continue

            for crop in crops:
                emb = get_embedding(crop)
                save_embedding(owner_uid, cat_id, emb)
                saved_count += 1

        except Exception as e:
            print(f"❌ image error: {e}")

    # done
    doc.reference.update({
        "status": "ready",
        "embeddingCount": saved_count,
        "updatedAt": firestore.SERVER_TIMESTAMP
    })

    print(f"✅ Done cat {cat_id}, embeddings: {saved_count}")


def run():
    docs = (
        db.collection(COLLECTION)
        .where("status", "==", "pending")
        .stream()
    )

    for doc in docs:
        process_one_document(doc)


if __name__ == "__main__":
    run()
