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
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")

# ---------------- Load Prediction Model (Random Forest) ----------------
prediction_model = None
try:
    pred_model_path = os.path.join(BASE_DIR, "stroke_model.pkl")
    print(f"Loading prediction model: {pred_model_path}")
    print(f"File exists: {os.path.exists(pred_model_path)}")
    prediction_model = joblib.load(pred_model_path)
    print("✅ Prediction model loaded!")
except Exception as e:
    print(f"❌ Prediction model failed: {e}")

# ---------------- Load Detection Model (SavedModel format) ----------------
detection_model = None
infer = None
try:
    saved_path = os.path.join(BASE_DIR, "stroke_cnn_savedmodel")
    print(f"Loading detection model: {saved_path}")
    print(f"Folder exists: {os.path.exists(saved_path)}")

    loaded = tf.saved_model.load(saved_path)
    infer = loaded.signatures["serving_default"]
    detection_model = infer
    print("✅ Detection model loaded via SavedModel!")
except Exception as e:
    print(f"❌ Detection model failed: {e}")

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
def prediction_page():
    return send_from_directory(BASE_DIR, "prediction.html")

@app.route("/detection.html")
def detection_page():
    return send_from_directory(BASE_DIR, "detection.html")

@app.route("/prediction.js")
def prediction_js():
    return send_from_directory(BASE_DIR, "prediction.js")

@app.route("/detection.js")
def detection_js():
    return send_from_directory(BASE_DIR, "detection.js")

# ---------------- Prediction API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if prediction_model is None:
        return jsonify({"success": False, "message": "Prediction model not loaded"}), 503
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        df = pd.DataFrame([{
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": int(data["smoking_status"])
        }])

        prob = prediction_model.predict_proba(df)[0][1]

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
        print(f"Prediction error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------- Detection API ----------------
@app.route("/detect", methods=["POST"])
def detect():
    if detection_model is None:
        return jsonify({"success": False, "message": "Detection model not loaded"}), 503
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image uploaded"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"success": False, "message": "Empty image file"}), 400

        filename = image_file.filename.replace(" ", "_")
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)

        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"success": False, "message": "Invalid image format"}), 400

        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict using SavedModel signature
        start_time = time.time()
        input_tensor = tf.constant(img)
        output = infer(input_tensor)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Get output value
        output_key = list(output.keys())[0]
        score = float(output[output_key].numpy().flatten()[0])

        return jsonify({
            "success": True,
            "result": "Stroke Detected" if score >= 0.5 else "Normal Brain",
            "confidence": score,
            "processing_ms": elapsed_ms
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------- Health Check ----------------
@app.route("/health", methods=["GET"])
def health():
    files_in_dir = os.listdir(BASE_DIR)
    return jsonify({
        "success": True,
        "prediction_model_loaded": prediction_model is not None,
        "detection_model_loaded": detection_model is not None,
        "tensorflow_version": tf.__version__,
        "files_found": files_in_dir
    })

# ---------------- Run Server ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)