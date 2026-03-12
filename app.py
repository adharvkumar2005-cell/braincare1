import os
import time
import cv2
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------- App Setup ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Load Prediction Model (Random Forest) ----------------
prediction_model = None
try:
    pred_model_path = os.path.join(BASE_DIR, "stroke_model.pkl")
    prediction_model = joblib.load(pred_model_path)
    app.logger.info("✅ Stroke prediction model loaded")
except Exception as e:
    app.logger.error(f"❌ Prediction model load failed: {e}")

# ---------------- Load Detection Model (CNN) ----------------
detection_model = None
try:
    det_model_path = os.path.join(BASE_DIR, "stroke_cnn_model.h5")
    detection_model = tf.keras.models.load_model(det_model_path)
    app.logger.info("✅ Stroke detection model loaded")
except Exception as e:
    app.logger.error(f"❌ Detection model load failed: {e}")

# ---------------- Home ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([{
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": int(data["smoking_status"])
        }])

        # Stroke probability (class = 1)
        prob = prediction_model.predict_proba(df)[0][1]

        # Medical thresholds (REALISTIC)
        if prob >= 0.30:
            risk = "High Stroke Risk"
        elif prob >= 0.15:
            risk = "Moderate Stroke Risk"
        else:
            risk = "Low Stroke Risk"

        return jsonify({
            "success": True,
            "result": risk,
            "risk_percentage": round(prob * 100, 2)
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
# ---------------- Detection API ----------------
@app.route("/detect", methods=["POST"])
def detect():
    if detection_model is None:
        return jsonify({
            "success": False,
            "message": "Detection model not available"
        }), 503

    try:
        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image uploaded"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"success": False, "message": "Empty image file"}), 400

        # Save image
        filename = image_file.filename.replace(" ", "_")
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"success": False, "message": "Invalid image format"}), 400

        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        start_time = time.time()
        prediction = detection_model.predict(img)
        elapsed_ms = int((time.time() - start_time) * 1000)

        pred_arr = np.asarray(prediction)
        if pred_arr.size == 1:
            score = float(pred_arr.flatten()[0])
        elif pred_arr.ndim == 2 and pred_arr.shape[1] == 2:
            score = float(pred_arr[0, 1])
        else:
            score = float(pred_arr.flatten()[0])

        return jsonify({
            "success": True,
            "result": "Stroke Detected" if score >= 0.5 else "Normal Brain",
            "confidence": float(score),
            "processing_ms": elapsed_ms
        })

    except Exception as e:
        app.logger.exception("Detection error")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# ---------------- Health Check ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "prediction_model_loaded": prediction_model is not None,
        "detection_model_loaded": detection_model is not None
    })

# ---------------- Serve Frontend Pages ----------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "login.html")

@app.route("/login.html")
def login():
    return send_from_directory(BASE_DIR, "login.html")

@app.route("/register.html")
def register():
    return send_from_directory(BASE_DIR, "register.html")

@app.route("/prediction.html")
def prediction():
    return send_from_directory(BASE_DIR, "prediction.html")

@app.route("/detection.html")
def detection():
    return send_from_directory(BASE_DIR, "detection.html")

# Serve JS files
@app.route("/prediction.js")
def prediction_js():
    return send_from_directory(BASE_DIR, "prediction.js")

@app.route("/detection.js")
def detection_js():
    return send_from_directory(BASE_DIR, "detection.js")

# ---------------- Run Server ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)