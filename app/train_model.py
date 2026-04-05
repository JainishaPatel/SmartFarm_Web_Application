import os
import pickle
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === Load environment variables ===
load_dotenv()
CSV_PATH = os.getenv("DATASET_PATH", "crop_data_india.csv")  # fallback if not set

# === Setup Project Paths ===
# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create models folder path
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create 'models' folder if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# === Load dataset ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ CSV file not found at {CSV_PATH}. Please check your .env or path.")

# === Read CSV file ===
df = pd.read_csv(CSV_PATH)

# === # Basic info ===
print(f"Loaded dataset from: {CSV_PATH}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# === Encode target column (Crop) ===
# Convert crop names into numbers
crop_encoder = LabelEncoder()
df['Crop'] = crop_encoder.fit_transform(df['Crop'])

# === Features & Target ===
X = df[['temperature', 'humidity']]   # Add more features later if needed
y = df['Crop']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Train model ===
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# === Model Evaluation ===
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=crop_encoder.classes_))

# === Save model & encoder ===
model_path = os.path.join(MODELS_DIR, "crop_simple_model.pkl")
encoder_path = os.path.join(MODELS_DIR, "crop_encoder.pkl")

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save encoder
with open(encoder_path, 'wb') as f:
    pickle.dump(crop_encoder, f)

print("Model and encoder saved successfully.")