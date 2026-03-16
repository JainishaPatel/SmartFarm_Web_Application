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

# === Load dataset ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ CSV file not found at {CSV_PATH}. Please check your .env or path.")

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset from: {CSV_PATH}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# === Encode target column (Crop) ===
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

# === Evaluate ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=crop_encoder.classes_))

# === Save model & encoder ===
with open('crop_simple_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(crop_encoder, f)

print("Model and encoder saved successfully.")