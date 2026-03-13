import tensorflow as tf
import numpy as np
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Load the old model
try:
    old_model = tf.keras.models.load_model("stroke_cnn_model.h5")
    print("✅ Old model loaded successfully")
    old_model.summary()
except Exception as e:
    print(f"❌ Could not load old model: {e}")
    print("Rebuilding model from scratch...")

    # Rebuild same architecture as train_cnn.py
    IMG = 128
    old_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG, IMG, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    old_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    print("✅ Model rebuilt with same architecture")

# Save in new keras format (compatible with all TF versions)
old_model.save("stroke_cnn_model.keras")
print("✅ Model saved as stroke_cnn_model.keras")

# Also save as SavedModel format as backup
old_model.save("stroke_cnn_model_saved")
print("✅ Model saved as SavedModel format")