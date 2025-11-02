
import pandas as pd
import os

DATA_PATH = os.path.join("data", "solar_data.csv")
CLEAN_PATH = os.path.join("data", "solar_clean.csv")

print("🧹 Cleaning data...")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove duplicates and missing values
df = df.dropna().drop_duplicates()

# Keep only relevant columns
columns = ["Temperature", "Humidity", "Wind Speed", "GHI"]
df = df[columns]

# Normalize column names (remove spaces)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Save cleaned dataset
df.to_csv(CLEAN_PATH, index=False)

print(f"✅ Clean data saved at: {CLEAN_PATH}")
