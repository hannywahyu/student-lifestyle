import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_path = 'drive/My Drive/Kaggle/best_stress_prediction_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title and description
st.title("Student Stress Level Predictor 🎓")
st.write("""
This app predicts your **stress level** based on your daily lifestyle activities such as study time, sleep, social interaction, and physical activities.
""")

# Input form
st.header("Enter Your Lifestyle Information")

study_hours = st.slider("Study Hours Per Day", 0.0, 10.0, 2.0)
extracurricular_hours = st.slider("Extracurricular Hours Per Day", 0.0, 5.0, 1.0)
sleep_hours = st.slider("Sleep Hours Per Day", 0.0, 12.0, 6.0)
social_hours = st.slider("Social Hours Per Day", 0.0, 6.0, 2.0)
physical_hours = st.slider("Physical Activity Hours Per Day", 0.0, 5.0, 1.0)
gpa = st.number_input("GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=3.0)

# Feature Engineering (same as in training)
study_sleep_interaction = study_hours * sleep_hours
physical_social_interaction = physical_hours * social_hours
gpa_squared = gpa ** 2

# Binning Study Hours (must match training code!)
study_bin = ''
if study_hours <= 2:
    study_bin = [1, 0, 0, 0]  # Low
elif study_hours <= 4:
    study_bin = [0, 1, 0, 0]  # Medium
elif study_hours <= 6:
    study_bin = [0, 0, 1, 0]  # High
else:
    study_bin = [0, 0, 0, 1]  # Very High

# Final feature vector
features = np.array([[
    study_hours,
    extracurricular_hours,
    sleep_hours,
    social_hours,
    physical_hours,
    gpa,
    study_sleep_interaction,
    physical_social_interaction,
    gpa_squared,
    *study_bin  # Unpack one-hot encoded bin
]])

# Prediction button
if st.button("Predict Stress Level"):
    prediction = model.predict(features)[0]
    st.subheader("Predicted Stress Level:")
    st.success(prediction)
