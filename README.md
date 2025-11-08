# ☀️ Solar Energy Prediction in India using Machine Learning

## 📘 Overview
This project aims to **predict solar energy output (in kWh)** for different regions of India using **machine learning techniques**.  
It uses weather-based features such as temperature, humidity, irradiance, and cloud cover to forecast daily solar energy generation.  

The model helps in:
- Estimating energy yield for solar plants,
- Understanding the impact of environmental conditions,
- Supporting renewable energy planning and optimization.

---

## 🧩 Dataset
The dataset is **synthetically generated** to simulate realistic solar energy data for major Indian cities.

**File:** `solar_energy_india_dataset.csv`  
**Records:** 1000 samples  
**Features:**
| Feature | Description |
|----------|--------------|
| Date | Date of measurement |
| Location | City in India (e.g., Jaipur, Chennai, Pune, etc.) |
| Latitude / Longitude | Geographical coordinates |
| Temperature (°C) | Average daily temperature |
| Humidity (%) | Relative humidity |
| Wind_Speed (m/s) | Daily average wind speed |
| Solar_Irradiance (W/m²) | Solar radiation intensity |
| Cloud_Cover (%) | Cloud percentage |
| Precipitation (mm) | Rainfall amount |
| Sunshine_Hours | Total sunlight duration per day |
| Month | Extracted month number |
| Season | Derived from month |
| Solar_Energy_Output (kWh) | **Target variable** – actual energy generated |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see Installation)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Solar_Energy_Prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib scikit-learn
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

---

## ⚙️ Model and Workflow

### 1. **Data Preprocessing**
- One-hot encoding for categorical columns (`Location`, `Season`)
- Feature-target split  
- Train-test division (80:20)
- Training samples: 800, Testing samples: 200

### 2. **Model Used**
A **Random Forest Regressor** from `scikit-learn` was chosen for its high performance on regression problems with non-linear relationships.

**Model Configuration:**
```python
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
```

### 3. **Model Evaluation**

The model achieved the following performance metrics:

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | 1.028 kWh |
| **Mean Squared Error (MSE)** | 1.522 |
| **Root Mean Squared Error (RMSE)** | 1.234 kWh |
| **R² Score** | 0.937 |

The **R² score of 0.937** indicates that the model explains approximately **93.7%** of the variance in solar energy output, demonstrating excellent predictive performance.

---

## 📊 Visualizations

The project includes two key visualizations:

1. **Actual vs Predicted Line Plot**: Compares actual and predicted solar energy output values, sorted for better visualization
2. **Residual Distribution Plot**: Shows the distribution of prediction errors (residuals) to assess model performance

---

## 🔮 Usage

### Running the Notebook

1. Open `Solar_Energy_Prediction.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The notebook will:
   - Load and preprocess the data
   - Train the Random Forest model
   - Evaluate model performance
   - Generate visualizations
   - Make future predictions

### Making Predictions

The model can predict solar energy output for new data points. Simply provide the required features (temperature, humidity, solar irradiance, etc.) and the model will output the predicted energy in kWh.

---

## 📁 Project Structure

```
Solar_Energy_Prediction/
│
├── Solar_Energy_Prediction.ipynb    # Main Jupyter notebook
├── solar_energy_india_dataset.csv   # Dataset file
└── README.md                        # Project documentation
```

---

## 🔍 Key Features

- **Robust Model**: Random Forest handles non-linear relationships and feature interactions
- **High Accuracy**: R² score of 0.937 indicates strong predictive capability
- **Comprehensive Analysis**: Includes data preprocessing, model training, evaluation, and visualization
- **Practical Application**: Can be used for real-world solar energy forecasting

---

## 📈 Results Summary

- **Model Performance**: Excellent (R² = 0.937)
- **Prediction Error**: Low (RMSE = 1.234 kWh)
- **Model Type**: Random Forest Regressor with 150 estimators
- **Data Split**: 80% training, 20% testing

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 👤 Author

Probal Sen

---

## 🙏 Acknowledgments

- Dataset: Synthetically generated for simulation purposes
- Libraries: pandas, numpy, matplotlib, scikit-learn
