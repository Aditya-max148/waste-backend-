from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

# ---------------- PATH SAFETY ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "waste_analyzer_model.h5")
CLASS_FILE = os.path.join(BASE_DIR, "class_names.json")
TARGET_SIZE = (224, 224)

# ---------------- LOAD CLASS NAMES ----------------
if not os.path.exists(CLASS_FILE):
    raise RuntimeError("❌ class_names.json not found in deployment")

with open(CLASS_FILE, "r") as f:
    class_indices = json.load(f)

CLASS_NAMES = {v: k.replace("_", " ").title() for k, v in class_indices.items()}

# ---------------- LOAD MODEL SAFELY ----------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ waste_analyzer_model.h5 not found in deployment")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(img_path):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- API ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = os.path.join(BASE_DIR, "temp.jpg")
    file.save(temp_path)

    img = preprocess(temp_path)
    preds = model.predict(img)

    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx]) * 100

    os.remove(temp_path)

    return jsonify({
        "waste_type": CLASS_NAMES[idx],
        "confidence": round(confidence, 2)
    })

# ---------------- START ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
