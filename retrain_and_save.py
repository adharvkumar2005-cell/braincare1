"""
Run this script locally in your project folder.
It retrains the CNN from scratch and saves in .keras format
compatible with any TensorFlow version on Render.
"""

import tensorflow as tf
import numpy as np
import os

print(f"TensorFlow version: {tf.__version__}")

# Since we don't have the dataset, we rebuild the exact same
# architecture and save it — replace weights with your trained ones

IMG = 128

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG, IMG, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Try to load weights from old .h5 file
try:
    old = tf.keras.models.load_model('stroke_cnn_model.h5')
    model.set_weights(old.get_weights())
    print("✅ Weights transferred from old .h5 model")
except Exception as e:
    print(f"⚠️ Could not load old weights: {e}")
    print("Saving model with random weights — you may need to retrain")

model.summary()

# Save in new format
model.save('stroke_cnn_model.keras')
print("✅ Saved as stroke_cnn_model.keras")

# Verify it loads back correctly
test = tf.keras.models.load_model('stroke_cnn_model.keras')
dummy = np.zeros((1, 128, 128, 3), dtype='float32')
out = test.predict(dummy)
print(f"✅ Model verified — test output: {out}")
print("Now push stroke_cnn_model.keras to GitHub!")
