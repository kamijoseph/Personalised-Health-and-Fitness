# webapplication

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="Personalized Health Recommendation System",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- STYLING ---------------------------
st.markdown("""
    <style>
    body {
        background-color: #000000 !important;
        color: #f5f5f5 !important;
    }
    .main {
        background-color: #111111;
        color: #f5f5f5;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.7);
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%);
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0083b0 0%, #00b4db 100%);
        transform: scale(1.03);
    }
    .title-text {
        color: #ffffff;
        text-shadow: 0 0 10px #00b4db;
    }
    .description-text {
        color: #cccccc;
        font-size: 1.1rem;
        text-align: justify;
    }
    .recommend-box {
        background-color: #0d1117;
        color: #f0f0f0;
        border-left: 4px solid #00b4db;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------- LOAD RESOURCES ---------------------------
@st.cache_resource
def load_resources():
    scaler = joblib.load("app/../resources/scaler.pkl")
    model = XGBClassifier()
    model.load_model("app/../resources/fitness_model.json")
    return scaler, model

scaler, model = load_resources()

# --------------------------- HEADER ---------------------------
st.markdown("<h1 class='title-text'>💪 Personalized Health Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='description-text'>This intelligent system predicts your <b>Health Class (A–D)</b> and provides personalized exercise and wellness recommendations based on your physiological metrics. Enter your information in the sidebar to begin.</p>", unsafe_allow_html=True)

st.markdown("---")

# --------------------------- SIDEBAR INPUT ---------------------------
st.sidebar.header("🧍 Input Your Details")

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

# --------------------------- INPUT PROCESSING ---------------------------
gender_val = 0 if gender == "F" else 1
input_data = np.array([
    age, gender_val, height, weight, body_fat,
    diastolic, systolic, grip, sit_bend, situps, broad_jump
]).reshape(1, -1)

input_scaled = scaler.transform(input_data)

# --------------------------- CENTERED BUTTON + IMAGE ---------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(
        "app/../resources/imggg.jpeg",   # your custom image path
        caption="Your health, quantified.",
        use_container_width=True          # make image full-width in column
    )

    st.markdown("<br>", unsafe_allow_html=True)  # spacing between image and button

    predict_button = st.button("🚀 Predict Health Class", use_container_width=True)

# --------------------------- PREDICTION ---------------------------
if predict_button:
    pred_class = model.predict(input_scaled)[0]
    class_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    st.markdown("---")
    st.subheader(f"🏋️ Predicted Health Class: **{class_map[pred_class]}**")

    recommendations = {
        "A": "🏆 **Excellent!** Maintain your current routine. Keep challenging your endurance and flexibility weekly.",
        "B": "💪 **Good!** Improve cardiovascular endurance. Add interval runs and compound lifts to your workouts.",
        "C": "⚖️ **Average.** Begin a structured 3-day training plan and adjust your nutrition for better body composition.",
        "D": "🩺 **Below average.** Seek a fitness coach or healthcare professional for a personalized recovery and training plan."
    }

    st.markdown("<div class='recommend-box'>" + recommendations[class_map[pred_class]] + "</div>", unsafe_allow_html=True)
    st.success("Prediction complete! Review your personalized recommendations above.")

# --------------------------- FOOTER ---------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and XGBoost")