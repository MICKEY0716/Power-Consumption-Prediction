import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# App Config
st.set_page_config(page_title="Power Consumption Predictor", page_icon="⚡", layout="wide")
st.title("⚡ Power Consumption Predictor - Zone 3")
st.markdown("Predict future power consumption using **Random Forest**, **SARIMA**, or **other ML models**.")

# Load Models
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_generalized_model.pkl")
    scaler = joblib.load("rf_scaler.pkl")
    return rf_model, scaler

rf_model, scaler = load_models()

# Sidebar - Model Selection
st.sidebar.header("Select Prediction Model")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Random Forest", "SARIMA")
)

st.sidebar.markdown("---")

# Random Forest Prediction
if model_choice == "Random Forest":
    st.subheader("Random Forest — Feature-Based Prediction")

    # Input Form
    with st.form(key="rf_input_form"):
        temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=10.0)
        pc_zone1 = st.number_input("PowerConsumption_Zone1 (kWh)", min_value=0.0, max_value=5000.0, value=1000.0)
        pc_zone2 = st.number_input("PowerConsumption_Zone2 (kWh)", min_value=0.0, max_value=5000.0, value=1000.0)
        hour = st.number_input("Hour of Day", min_value=0, max_value=23, value=12)
        dayofweek = st.number_input("Day of Week", min_value=0, max_value=6, value=2)
        is_weekend = st.selectbox("Is Weekend?", [0, 1])
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        
        submitted = st.form_submit_button("Predict PowerConsumption_Zone3")
    
    if submitted:
        # Prepare input
        input_data = np.array([[temperature, humidity, windspeed, pc_zone1, pc_zone2,
                                hour, dayofweek, is_weekend, month]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]
        st.success(f"Predicted Power Consumption (Zone 3): **{prediction:.2f} kWh**")

        # Feature Importance Plot
        st.markdown("### Feature Importance")
        importances = rf_model.feature_importances_
        features = ["Temperature","Humidity","WindSpeed","PC_Zone1","PC_Zone2","Hour","DayOfWeek","Is_Weekend","Month"]
        feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color="teal")
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        st.pyplot(fig)

# SARIMA Forecast
elif model_choice == "SARIMA":
    st.subheader("SARIMA — Time Series Forecast")

    st.markdown("""
    SARIMA forecasts future power consumption **based on historical values** of Zone 3.
    """)

    # Upload historical dataset
    uploaded_file = st.file_uploader("Upload CSV (with 'Datetime' & 'PowerConsumption_Zone3')", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="latin1")
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").set_index("Datetime")
        y = df["PowerConsumption_Zone3"].resample("D").mean().dropna()

        forecast_days = st.number_input("Forecast Next N Days", min_value=1, max_value=365, value=30)
        if st.button("Run SARIMA Forecast"):
            st.info("⏳ Fitting SARIMA model, please wait...")

            # Fit SARIMA (stable parameters)
            model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,7),
                            enforce_stationarity=False, enforce_invertibility=False)
            sarima_fit = model.fit(disp=False)

            forecast = sarima_fit.forecast(steps=forecast_days)

            # Metrics if desired (on last 20% as test)
            split_idx = int(len(y)*0.8)
            train, test = y[:split_idx], y[split_idx:]
            sarima_model_test = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7),
                                        enforce_stationarity=False, enforce_invertibility=False)
            sarima_fit_test = sarima_model_test.fit(disp=False)
            forecast_test = sarima_fit_test.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, forecast_test))
            mae = mean_absolute_error(test, forecast_test)
            r2 = r2_score(test, forecast_test)

            st.success(f"Forecast for next {forecast_days} days completed!")
            st.markdown(f"**Test Set Metrics:** RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.4f}")

            # Plot forecast
            fig = px.line(title="SARIMA Forecast vs Historical")
            fig.add_scatter(x=y.index, y=y.values, mode='lines', name="Historical")
            fig.add_scatter(x=forecast.index, y=forecast.values, mode='lines', name="Forecast")
            fig.update_layout(xaxis_title="Date", yaxis_title="Power Consumption (kWh)")
            st.plotly_chart(fig, use_container_width=True)
