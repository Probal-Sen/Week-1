import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("models", "solar_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Example new data point [Temperature, Humidity, Wind Speed]
sample = np.array([[33, 40, 10]])

prediction = model.predict(sample)
print(f"🔆 Predicted Solar Irradiance (GHI): {prediction[0]:.2f}")
