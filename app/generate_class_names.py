import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
data_dir = os.getenv("PLANT_DATASET_PATH")

# Load the already trained model
model = load_model("plant_disease_model.h5")

# Extract class names from the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128,128),
    batch_size=32
)

class_names = train_ds.class_names
print("Class names:", class_names)

# Save class names to pickle for Flask app
with open("plant_class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("plant_class_names.pkl created successfully!")
