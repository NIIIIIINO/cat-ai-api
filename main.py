import os
import numpy as np
import cv2
from flask import Flask, request, jsonify

from embeddings import get_embedding
from identify_cat import identify_cat

# ================= CONFIG =================
USE_YOLO_DEFAULT = True
YOLO_MODEL_PATH = "yolov8n.pt"

app = Flask(__name__)

# ================= YOLO (lazy) =================
_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("⏳ Loading YOLO...")
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO ready")
    return _yolo_model


# ================= utils =================
def read_image(file) -> np.ndarray:
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_cats(image: np.ndarray):
    """
    return list of cropped RGB images
    """
    yolo = get_yolo()
    results = yolo(image, conf=0.4, verbose=False)
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


def load_user_cat_bank(user_uid: str):
    """
    TODO: ดึง embedding จาก Firestore / GCS
    structure:
    {
        cat_uid: [emb1, emb2, ...]
    }
    """
    return {}  # ← เสียบ logic จริงตรงนี้


# ================= routes =================
@app.route("/", methods=["GET"])
def health():
    return "OK"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    user_uid = request.form.get("user_uid")
    if not user_uid:
        return jsonify({"error": "user_uid required"}), 400

    use_yolo = request.form.get(
        "use_yolo",
        str(USE_YOLO_DEFAULT)
    ).lower() == "true"

    try:
        image = read_image(request.files["image"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    cat_bank = load_user_cat_bank(user_uid)
    if not cat_bank:
        return jsonify({"error": "no cats registered for this user"}), 404

    crops = detect_cats(image) if use_yolo else [image]

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
        "user_uid": user_uid,
        "count": len(results),
        "results": results
    })


# ❌ ไม่ใช้ app.run() ตอน deploy
# gunicorn จะเป็นคนเรียก
