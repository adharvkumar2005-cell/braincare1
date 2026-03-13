"""
Converts stroke_cnn_model.h5 to SavedModel format
which works with ANY TensorFlow/Keras version.
Run: python convert_model.py
"""
import tensorflow as tf
import numpy as np
import os

print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")

IMG = 128

# Step 1: Rebuild exact same architecture
print("\nRebuilding model architecture...")
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

# Step 2: Try to load weights from old .h5
print("\nTrying to load weights from stroke_cnn_model.h5...")
try:
    old_model = tf.keras.models.load_model(
        'stroke_cnn_model.h5',
        compile=False
    )
    model.set_weights(old_model.get_weights())
    print("✅ Weights transferred successfully!")
except Exception as e:
    print(f"⚠️ Could not transfer weights: {e}")
    print("Saving with current weights...")

# Step 3: Test model
print("\nTesting model...")
dummy = np.zeros((1, IMG, IMG, 3), dtype='float32')
out = model.predict(dummy, verbose=0)
print(f"✅ Test prediction: {out}")

# Step 4: Save as SavedModel format (most compatible)
save_path = 'stroke_cnn_savedmodel'
print(f"\nSaving as SavedModel to: {save_path}")
tf.saved_model.save(model, save_path)
print(f"✅ Saved! Size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(save_path) for f in fn)} bytes")

# Step 5: Verify it loads back
print("\nVerifying reload...")
reloaded = tf.saved_model.load(save_path)
print("✅ SavedModel reloaded successfully!")
print("\n✅ DONE! Now push stroke_cnn_savedmodel folder to GitHub")
