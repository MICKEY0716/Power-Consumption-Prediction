# ⚡ Power Consumption Prediction

> **End-to-End Time Series Forecasting Project using SARIMA**

---

## 📌 Overview

Accurate power consumption forecasting is critical for efficient energy planning, load management, and sustainable infrastructure development.  
This project focuses on predicting future power consumption using historical electricity usage data by leveraging **time-series analysis and seasonal forecasting techniques**.

The solution is designed as a **complete pipeline** — from data analysis and modeling to deployment through an interactive user interface.

---

## 🎯 Problem Statement

Electricity demand varies significantly due to:
- Time-based usage patterns
- Seasonal effects
- Long-term consumption trends

Without reliable forecasting models, power planners may face:
- Overproduction or wastage
- Energy shortages
- Inefficient grid utilization

**Objective:**  
Build a robust and interpretable forecasting model that can accurately predict future power consumption based on historical patterns.

---

## 📊 Dataset Overview

- Historical power consumption data  
- Time-indexed observations  
- Contains clear trends and seasonal components  

> Dataset details are intentionally kept concise to focus on modeling and forecasting methodology.

---

## 🧠 Approach & Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Trend and seasonality visualization
- Missing value handling
- Stationarity checks using statistical tests
- Seasonal decomposition

### 2️⃣ Feature Engineering
- Time-based transformations
- Lag and rolling window analysis
- Seasonal pattern extraction

### 3️⃣ Model Selection
- Evaluated classical time-series models
- Selected **SARIMA** due to:
  - Strong seasonal handling
  - Interpretability
  - Stability on time-dependent data

### 4️⃣ Model Training & Forecasting
- Hyperparameter tuning
- Model validation
- Short-term and long-term demand forecasting

---

## 🤖 Model Used

### 🔹 SARIMA (Seasonal ARIMA)

SARIMA was chosen because it effectively captures:
- Autoregressive behavior
- Moving averages
- Seasonal cycles
- Long-term trends

This makes it well-suited for **real-world power consumption forecasting**.

---

## 📈 Results & Insights

- Forecast closely aligns with historical consumption trends
- Seasonal patterns are captured effectively
- Model produces smooth and stable predictions
- Results are interpretable and suitable for decision-making

📌 Detailed plots and evaluations are available in the notebook.

---

## 🖥️ User Interface

An interactive user interface is included to enhance usability and interpretation.

### UI Capabilities:
- Visualize historical power consumption
- Display future forecasts
- Make predictions accessible to non-technical users

This bridges the gap between **model development and real-world usability**.

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling:** Statsmodels (SARIMA)  
- **Deployment:** Streamlit / Flask  
- **Model Persistence:** Pickle  

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
---

### Step 2: Run the Application
```bash
python ui/app.py
```
