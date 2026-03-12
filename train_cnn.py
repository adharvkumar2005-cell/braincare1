from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]

IMG = 128

gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = gen.flow_from_directory(
    "dataset",
    target_size=(IMG, IMG),
    class_mode="binary",
    subset="training"
)

val = gen.flow_from_directory(
    "dataset",
    target_size=(IMG, IMG),
    class_mode="binary",
    subset="validation"
)

model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(IMG,IMG,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train, validation_data=val, epochs=5)

model.save("stroke_cnn_model.h5")

print("Detection model created")