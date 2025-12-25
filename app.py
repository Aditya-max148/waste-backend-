from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)  # allow React to call Flask

MODEL_PATH = "waste_analyzer_model.h5"
CLASS_FILE = "class_names.json"
TARGET_SIZE = (224, 224)

# 🔴 MUST MATCH TRAINING OUTPUT
# ---------------- LOAD CLASS NAMES ----------------
if not os.path.exists(CLASS_FILE):
    print("❌ class_names.json not found. Train the model first.")
    exit()

with open(CLASS_FILE, "r") as f:
    class_indices = json.load(f)
CLASS_NAMES = {v: k.replace("_", " ").title() for k, v in class_indices.items()}

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(img_path):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    filepath = "temp.jpg"
    file.save(filepath)

    img = preprocess(filepath)
    preds = model.predict(img)

    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx]) * 100

    os.remove(filepath)

    return jsonify({
        "waste_type": CLASS_NAMES[idx],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
