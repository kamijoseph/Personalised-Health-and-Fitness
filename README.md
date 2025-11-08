# Personalized Health Recommendation System

## Problem Statement
This project predicts an individual’s health class (A-D) based on body metrics and provides tailored recommendations.

---

## 🚀 Live Demo

Experience the app in action here:  
👉 [Personalized Health Recommendation System (Live Demo)](https://personalised-health-and-fitness.streamlit.app/)

*(Click the link above to open the interactive web app.)*

---

## Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)
- Columns: age, gender, height_cm, weight_kg, body_fat_%, diastolic, systolic, gripForce, sit and bend forward_cm, sit-ups, broad jump_cm, class

## EDA Insights
- Correlations between BMI, body fat %, and strength measures
- Class distribution is stratified (A-D)
- Visualizations: histograms, boxplots, correlation heatmap

## Modeling
- Models tried: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Evaluation: Accuracy, F1-score, Confusion Matrix
- Best model: (to be filled after evaluation)

## Deployment
- Streamlit app for user input
- Predicts class and displays recommendations

## Recommendations
- A: Excellent
- B: Good
- C: Average
- D: Below Average
