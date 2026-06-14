import streamlit as st
import pandas as pd
import pickle
import os
from huggingface_hub import hf_hub_download

# set_page_config MUST be the first Streamlit command
st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="🔧")

# Read repo names from environment variables
HF_USERNAME   = os.getenv("HF_USERNAME",   "gowdhamankarthikeyan")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "engine-maintenance-predictor")

# Load model from HuggingFace Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=f"{HF_USERNAME}/{HF_MODEL_NAME}",
        filename="best_model.pkl"
    )
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

FEATURE_COLS = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temp', 'Coolant_Temp',
    'rpm_x_fuel_pressure', 'oil_health_index', 'rpm_bins'
]
THRESHOLD = 0.5

def engineer_features(df):
    """Apply the same feature engineering used during model training."""
    df = df.copy()
    df['rpm_x_fuel_pressure'] = df['Engine_RPM'] * df['Fuel_Pressure']
    df['oil_health_index'] = df['Lub_Oil_Pressure'] / df['Lub_Oil_Temp']
    df['rpm_bins'] = pd.cut(
        df['Engine_RPM'],
        bins=[0, 500, 1000, float('inf')],
        labels=[0, 1, 2]
    ).astype(int)
    return df

st.title("🔧 Engine Predictive Maintenance")
st.write("Predict whether an engine requires maintenance based on sensor readings.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("🔬 Sensor Readings")
    st.write("Enter the six raw sensor values. Engineered features are computed automatically.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Engine Performance**")
        engine_rpm    = st.number_input("Engine RPM",          min_value=0.0, value=800.0, step=10.0)
        fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, value=6.5,   step=0.1)
    with col2:
        st.markdown("**Lubrication System**")
        lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, value=3.3,  step=0.1)
        lub_oil_temp     = st.number_input("Lub Oil Temp (°C)",      min_value=0.0, value=77.0, step=0.5)
    with col3:
        st.markdown("**Cooling System**")
        coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, value=2.3,  step=0.1)
        coolant_temp     = st.number_input("Coolant Temp (°C)",      min_value=0.0, value=78.0, step=0.5)

    if st.button("Predict Maintenance Need"):
        input_df = pd.DataFrame([{
            'Engine_RPM':       engine_rpm,
            'Lub_Oil_Pressure': lub_oil_pressure,
            'Fuel_Pressure':    fuel_pressure,
            'Coolant_Pressure': coolant_pressure,
            'Lub_Oil_Temp':     lub_oil_temp,
            'Coolant_Temp':     coolant_temp
        }])
        input_df = engineer_features(input_df)
        prob = model.predict_proba(input_df[FEATURE_COLS])[0, 1]
        pred = int(prob >= THRESHOLD)

        st.markdown("---")
        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"⚠️ Maintenance Required (failure probability: {prob:.2%})")
            st.write("The engine is showing signs of degradation. Schedule maintenance.")
        else:
            st.success(f"✅ Normal Operation (failure probability: {prob:.2%})")
            st.write("The engine is operating within normal parameters.")

with tab2:
    st.subheader("📂 Batch Prediction")
    st.write("Upload a CSV with columns: Engine_RPM, Lub_Oil_Pressure, Fuel_Pressure, "
             "Coolant_Pressure, Lub_Oil_Temp, Coolant_Temp")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_df = engineer_features(batch_df)
        probs    = model.predict_proba(batch_df[FEATURE_COLS])[:, 1]
        preds    = (probs >= THRESHOLD).astype(int)

        batch_df['failure_probability'] = probs.round(4)
        batch_df['prediction'] = ['Maintenance' if p == 1 else 'Normal' for p in preds]

        maint_count  = int(preds.sum())
        normal_count = len(preds) - maint_count
        st.warning(f"⚠️ Maintenance Required: {maint_count} of {len(preds)}")
        st.success(f"✅ Normal Operation: {normal_count} of {len(preds)}")
        st.dataframe(batch_df)

        csv_out = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv_out, "predictions.csv", "text/csv")
