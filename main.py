import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

from embeddings import get_embedding
from identify_cat import identify_cat

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.4

app = Flask(__name__)

# ---------- LAZY YOLO ----------
yolo_model = None

def get_yolo():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        print("⏳ Loading YOLO...")
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
    return {}


# ---------- ROUTES ----------
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

    try:
        image = read_image(request.files["image"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    crops = detect_cat_crops(image)
    if not crops:
        return jsonify({"status": "no_cat_detected"})

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
