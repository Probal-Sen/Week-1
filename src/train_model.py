import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

DATA_PATH = os.path.join("data", "solar_clean.csv")
MODEL_PATH = os.path.join("models", "solar_model.pkl")

print("⚙️ Training solar energy prediction model...")

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Define features and target
X = df[["Temperature", "Humidity", "Wind_Speed"]]
y = df["GHI"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"✅ Training complete!")
print(f"📈 R² Score: {r2:.3f}")
print(f"📉 Mean Squared Error: {mse:.3f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved at: {MODEL_PATH}")
