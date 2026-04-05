import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from dotenv import load_dotenv

# === Load Environment Variables ===
load_dotenv()
dataset_path = os.getenv("SOIL_DATASET_PATH", "datasets/crop_recommendation.csv")

# === Setup Models Folder ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

# === Check Dataset ===
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

# === Load dataset ===
df = pd.read_csv(dataset_path)

# ---------------------------
# 1) Create Soil Health Label
# ---------------------------
def soil_health(row):
    if 80 <= row['N'] <= 150 and 30 <= row['P'] <= 70 and 30 <= row['K'] <= 70 and 6 <= row['ph'] <= 7:
        return 'Healthy'
    elif 50 <= row['N'] < 80 or 15 <= row['P'] < 30 or 15 <= row['K'] < 30 or 5.5 <= row['ph'] < 6 or 7 < row['ph'] <= 7.5:
        return 'Moderate'
    else:
        return 'Poor'

df['soil_health'] = df.apply(soil_health, axis=1)

# Common Features
X = df[['N','P','K','temperature','humidity','ph']]

# ---------------------------
# 2) Soil Health Model
# ---------------------------
y_health = df['soil_health']

X_train, X_test, y_train, y_test = train_test_split(X, y_health, test_size=0.2, random_state=42)

health_model = RandomForestClassifier()
health_model.fit(X_train, y_train)

# === Save soil health model ===
health_model_path = os.path.join(MODELS_DIR, "soil_health_model.pkl")

with open(health_model_path, "wb") as f:
    pickle.dump(health_model, f)

# ---------------------------
# 3) Crop Recommendation Model
# ---------------------------
y_crop = df['label']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_crop, test_size=0.2, random_state=42)

crop_model = RandomForestClassifier()
crop_model.fit(X_train2, y_train2)

# Save crop model
crop_model_path = os.path.join(MODELS_DIR, "crop_full_model.pkl")

with open(crop_model_path, "wb") as f:
    pickle.dump(crop_model, f)

print("Both models trained and saved successfully!")
