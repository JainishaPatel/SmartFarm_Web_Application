import os
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

load_dotenv()

# Path to your dataset
dataset_path = os.getenv("SOIL_ANALYSIS_DATASET_PATH")

# Image preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 20% validation
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
num_classes = len(train_data.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=15)

# Save the model
model.save('soil_model.h5')

print("Model trained and saved as soil_model.h5")
print("Class mapping:", train_data.class_indices)
