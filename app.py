import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.utils import plot_feature_importance
from src.preprocess import preprocess_and_scale
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Smart Predictive Maintenance System", layout="wide")

# --- CSS to style the app ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #aaaaaa;
        text-transform: uppercase;
    }
    .alert-danger {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Smart Predictive Maintenance System")
st.markdown("Real-time inference dashboard for machine health monitoring and remaining useful life (RUL) estimation.")

# --- Load Models & Scaler ---
@st.cache_resource
def load_models():
    base_path = 'models'
    classifier = joblib.load(os.path.join(base_path, 'classifier.pkl'))
    rul_model = joblib.load(os.path.join(base_path, 'rul_model.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(base_path, 'feature_columns.pkl'))
    return classifier, rul_model, scaler, feature_cols

try:
    classifier, rul_model, scaler, feature_cols = load_models()
except FileNotFoundError:
    st.error("Models not found. Please run the training scripts first.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Machine Sensor Inputs")
machine_type = st.sidebar.selectbox("Machine Type", options=['L', 'M', 'H'], index=0)
air_temp = st.sidebar.slider("Air temperature [K]", min_value=250.0, max_value=350.0, value=298.1)
process_temp = st.sidebar.slider("Process temperature [K]", min_value=250.0, max_value=350.0, value=308.6)
rot_speed = st.sidebar.slider("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1551)
torque = st.sidebar.slider("Torque [Nm]", min_value=10.0, max_value=100.0, value=42.8)
tool_wear = st.sidebar.slider("Tool wear [min]", min_value=0, max_value=300, value=0)

# Create input dataframe
input_data = pd.DataFrame([{
    'Type': machine_type,
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': rot_speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear
}])

# --- Inference ---
# Manually apply preprocessing to single row
def process_input(df, scaler, feature_cols):
    df_engineered = df.copy()
    # Remove brackets
    df_engineered.columns = [col.replace('[', '').replace(']', '').replace('<', '').strip() for col in df_engineered.columns]
    
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df_engineered['Type'] = df_engineered['Type'].map(type_mapping)
    df_engineered['temp_diff'] = df_engineered['Process temperature K'] - df_engineered['Air temperature K']
    df_engineered['power'] = df_engineered['Torque Nm'] * df_engineered['Rotational speed rpm']
    df_engineered['wear_rate'] = df_engineered['Tool wear min'] / df_engineered['Rotational speed rpm']
    
    features_to_scale = [
        'Air temperature K', 'Process temperature K', 
        'Rotational speed rpm', 'Torque Nm', 'Tool wear min',
        'temp_diff', 'power', 'wear_rate'
    ]
    df_engineered[features_to_scale] = scaler.transform(df_engineered[features_to_scale])
    
    # Ensure correct column order
    return df_engineered[feature_cols]

processed_input = process_input(input_data, scaler, feature_cols)

# Predict Failure
failure_prob = classifier.predict_proba(processed_input)[0][1]
failure_pred = classifier.predict(processed_input)[0]

# Predict RUL
rul_pred = rul_model.predict(processed_input)[0]

# Calculate Health Score (0-100) (Removed)

# --- Dashboard Display ---

# Alert Banner
if failure_prob > 0.6:
    st.markdown('<div class="alert-danger">CRITICAL ALERT: High Probability of Machine Failure Detected!</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    color = "#ff4b4b" if failure_pred == 1 else "#4CAF50"
    status_text = "FAILURE IMMINENT" if failure_pred == 1 else "HEALTHY"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Failure Prediction</div>
        <div class="metric-value" style="color: {color};">{status_text}</div>
        <div style="font-size: 14px; margin-top: 5px;">Probability: {failure_prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimated RUL</div>
        <div class="metric-value" style="color: #2196F3;">{rul_pred:.0f} mins</div>
        <div style="font-size: 14px; margin-top: 5px;">Remaining Useful Life</div>
    </div>
    """, unsafe_allow_html=True)

# --- Optional Toggles ---
st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    if st.checkbox("Show Feature Importance (Classifier)"):
        st.subheader("Random Forest Feature Importance")
        # Ensure it's a random forest or xgboost to have feature importance
        if hasattr(classifier, 'feature_importances_'):
            fig = plot_feature_importance(classifier, feature_cols, title="")
            st.pyplot(fig)
        else:
            st.info("Current model does not support feature importance out of the box.")

with col_b:
    if st.checkbox("Show Raw Dataset Preview"):
        st.subheader("Historical Data Preview")
        try:
            df_preview = pd.read_csv('data/ai4i2020.csv').head(10)
            st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.error("Could not load dataset preview.")
