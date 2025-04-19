import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load the model and encoders (assume saved earlier)
model = xgb.XGBClassifier()
model.load_model("C:\\Users\\Deependra\\fertilizer_projects\\models\\fertilizer_model.json")

with open("C:\\Users\\Deependra\\fertilizer_projects\\models\\encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("C:\\Users\\Deependra\\fertilizer_projects\\models\\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

soil_encoder = encoders['Soil Type']
crop_encoder = encoders['Crop Type']
fertilizer_encoder = encoders['Fertilizer Name']

st.title("🌱 Fertilizer Recommendation System")

# Input fields
temperature = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 10, 100, 50)
moisture = st.slider("Moisture (%)", 0, 100, 30)
soil_type = st.selectbox("Soil Type", soil_encoder.classes_)
crop_type = st.selectbox("Crop Type", crop_encoder.classes_)
nitrogen = st.slider("Nitrogen Level", 0, 140, 30)
potassium = st.slider("Potassium Level", 0, 140, 30)
phosphorous = st.slider("Phosphorous Level", 0, 140, 30)

if st.button("Predict Fertilizer"):
    # Prepare input
    input_data = pd.DataFrame([[
        temperature,
        humidity,
        moisture,
        soil_encoder.transform([soil_type])[0],
        crop_encoder.transform([crop_type])[0],
        nitrogen,
        potassium,
        phosphorous
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    fertilizer_name = fertilizer_encoder.inverse_transform([prediction])[0]

    st.success(f"🌾 Recommended Fertilizer: **{fertilizer_name}**")
