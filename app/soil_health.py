import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from dotenv import load_dotenv

load_dotenv()
dataset_path = os.getenv("SOIL_DATASET_PATH", "datasets/crop_recommendation.csv")

# Load dataset
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

with open("soil_health_model.pkl", "wb") as f:
    pickle.dump(health_model, f)

# ---------------------------
# 3) Crop Recommendation Model
# ---------------------------
y_crop = df['label']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_crop, test_size=0.2, random_state=42)

crop_model = RandomForestClassifier()
crop_model.fit(X_train2, y_train2)

with open("crop_full_model.pkl", "wb") as f:
    pickle.dump(crop_model, f)

print("Both models trained and saved successfully!")
