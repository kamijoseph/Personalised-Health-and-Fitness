
# webapplication

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# loading resources
@st.cache_resource
def load_resources():
    # loading encoder
    encoder = joblib.load("app/../resources/encoder.pkl")
    
    # laoding scaler
    scaler = joblib.load("app/../resources/scaler.pkl")

    # loading model
    model = XGBClassifier()
    model.load_model("app/../resources/fitness_model.json")

    return encoder, scaler, model

# encoder, scaler and model
encoder, scaler, model = load_resources()

st.title("Personalized Health Recommendation System")

# Sidebar input
age = st.sidebar.slider("Age", 20, 64, 30)
gender = st.sidebar.selectbox("Gender", ["F", "M"])
height = st.sidebar.number_input("Height (cm)", 140, 210, 170)
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)
body_fat = st.sidebar.slider("Body Fat %", 5, 50, 20)
diastolic = st.sidebar.slider("Diastolic BP", 60, 120, 80)
systolic = st.sidebar.slider("Systolic BP", 90, 180, 120)
grip = st.sidebar.slider("Grip Force", 10, 80, 40)
sit_bend = st.sidebar.slider("Sit & Bend Forward (cm)", 0, 50, 20)
situps = st.sidebar.slider("Sit-ups", 0, 60, 30)
broad_jump = st.sidebar.slider("Broad Jump (cm)", 50, 300, 150)

# Preprocess input
gender_val = 0 if gender=="F" else 1
BMI = weight / ((height/100)**2)
bp_diff = systolic - diastolic

input_data = np.array([age, height, weight, body_fat, diastolic, systolic,
                       grip, sit_bend, situps, broad_jump, gender_val, BMI, bp_diff]).reshape(1, -1)

input_scaled = scaler.transform(input_data)

# Prediction
pred_class = model.predict(input_scaled)[0]
class_map = {0:"A", 1:"B", 2:"C", 3:"D"}
st.subheader(f"Predicted Health Class: {class_map[pred_class]}")

# Recommendations
recommendations = {
    "A": "Excellent! Maintain your current fitness routine.",
    "B": "Good! Focus on improving core strength and cardiovascular health.",
    "C": "Average. Consider structured exercise and balanced diet.",
    "D": "Below average. Consult a healthcare professional and start a fitness plan."
}
st.write(recommendations[class_map[pred_class]])
