import os
import numpy as np
import json
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# === Load Environment Variables ===
load_dotenv()

# === Path to your dataset ===
dataset_path = os.getenv("SOIL_ANALYSIS_DATASET_PATH")

# === Setup Models Folder ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

# === Check Dataset Path ===
if not dataset_path or not os.path.exists(dataset_path):
    raise FileNotFoundError("❌ Dataset path is invalid. Check SOIL_ANALYSIS_DATASET_PATH in .env")


# === Image preprocessing ===
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

# === Build CNN Model ===
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

# === Train the model ===
model.fit(train_data, validation_data=val_data, epochs=15)

# === Save the model ===
model_path = os.path.join(MODELS_DIR, "soil_model.h5")

model.save(model_path)

print("Model trained and saved as soil_model.h5")

# === Mapping ===
class_map_path = os.path.join(MODELS_DIR, "soil_class_map.json")

with open(class_map_path, "w") as f:
    json.dump(train_data.class_indices, f)

print(f"📊 Class mapping saved at: {class_map_path}")

# === Reverse mapping (index → class name) ===
reverse_class_map = {v: k for k, v in train_data.class_indices.items()}

reverse_map_path = os.path.join(MODELS_DIR, "soil_class_map_reverse.json")

with open(reverse_map_path, "w") as f:
    json.dump(reverse_class_map, f)

print(f"🔁 Reverse class mapping saved at: {reverse_map_path}")