import pandas as pd
from dotenv import load_dotenv
import os

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
csv_path = os.getenv("PRICES_DATASET_PATH")

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv(csv_path)
print("Original shape:", df.shape)

# ---------------------------
# Keep required columns
# ---------------------------
required_cols = ["STATE", "District Name", "Commodity", "Modal_Price", "Date"]
df = df[required_cols]
print("After keeping columns:", df.shape)

# ---------------------------
# Convert Date column
# ---------------------------
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.strftime('%d-%b-%Y')

# ---------------------------
# Keep last 1 year only
# ---------------------------
df = df[df["Date"] >= "2024-01-01"]
print("After date filter:", df.shape)

# ---------------------------
# Keep ONE latest row per (STATE + DISTRICT + COMMODITY)
# ---------------------------
df_unique = (
    df
    .sort_values("Date")  # ensures latest date is last
    .groupby(["STATE", "District Name", "Commodity"], as_index=False)
    .last()
)
print("After unique grouping:", df_unique.shape)

# ---------------------------
# Sort for clean output
# ---------------------------
df_unique = df_unique.sort_values(["STATE", "District Name", "Commodity"])

# ---------------------------
# Convert Modal_Price to per kg
# Assuming original price is per quintal (100 kg)
# ---------------------------
df_unique["Price_per_kg"] = (df_unique["Modal_Price"] / 100).round(2)
df_unique["Price_display"] = df_unique["Price_per_kg"].apply(lambda x: f"₹ {x} / kg")

# ---------------------------
# Save cleaned dataset
# ---------------------------
output_file = "market_price_unique.csv"
df_unique.to_csv(output_file, index=False)
print(f"Saved cleaned data as {output_file}")
