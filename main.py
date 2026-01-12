import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

from embeddings import get_embedding
from identify_cat import identify_cat

# ---------- CONFIG ----------
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.4

app = Flask(__name__)

# ---------- LAZY YOLO (สำคัญมากสำหรับ Cloud Run) ----------
yolo_model = None

def get_yolo():
    """
    โหลด YOLO ตอนมี request ครั้งแรกเท่านั้น
    เพื่อให้ Cloud Run เปิด port ทัน
    """
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        print("⏳ Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded")
    return yolo_model


# ---------- UTILS ----------
def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_cat_crops(image):
    """
    return list of cropped RGB images
    """
    model = get_yolo()
    results = model(image, conf=YOLO_CONF, verbose=False)

    crops = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] != "cat":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    return crops


def load_user_cat_bank(user_uid):
    """
    TODO:
    ดึง embeddings ของแมวจาก Cloud Storage / Firestore
    ตอนนี้เว้นไว้ก่อน
    """
    return {}   # {cat_uid: [emb1, emb2, ...]}


# ---------- ROUTES ----------
@app.route("/", methods=["GET"])
def health():
    return "OK"


@app.route("/predict", methods=["POST"])
def predict():
    # --- check input ---
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    user_uid = request.form.get("user_uid")
    if not user_uid:
        return jsonify({"error": "user_uid required"}), 400

    try:
        image = read_image(request.files["image"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # --- YOLO ---
    crops = detect_cat_crops(image)
    if not crops:
        return jsonify({"status": "no_cat_detected"})

    # --- load user cat embeddings ---
    cat_bank = load_user_cat_bank(user_uid)
    if not cat_bank:
        return jsonify({"status": "no_cat_registered"})

    results = []
    for idx, crop in enumerate(crops):
        emb = get_embedding(crop)
        cat_uid, score = identify_cat(emb, cat_bank)

        results.append({
            "index": idx,
            "cat_uid": cat_uid,
            "score": float(score)
        })

    return jsonify({
        "status": "ok",
        "count": len(results),
        "results": results
    })


# ---------- ENTRYPOINT ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
