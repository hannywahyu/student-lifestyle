import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model
print(os.listdir())  # Ini akan menampilkan semua file di direktori kerja saat ini
with open('drive/My Drive/Colab Notebooks/best_stress_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("🎓 Prediksi Tingkat Stres Mahasiswa")

st.markdown("Masukkan informasi gaya hidup harian mahasiswa untuk memprediksi tingkat stres.")

# Input fields
study_hours = st.slider("Jam Belajar per Hari", 0.0, 12.0, 2.0)
sleep_hours = st.slider("Jam Tidur per Hari", 0.0, 12.0, 6.0)
physical_activity_hours = st.slider("Jam Aktivitas Fisik per Hari", 0.0, 5.0, 1.0)
social_hours = st.slider("Jam Bersosialisasi per Hari", 0.0, 8.0, 2.0)
extracurricular_hours = st.slider("Jam Ekstrakurikuler per Hari", 0.0, 5.0, 1.0)
gpa = st.number_input("IPK (GPA)", min_value=0.0, max_value=4.0, value=3.0)

# Feature Engineering (sama seperti yang digunakan saat training)
study_sleep_interaction = study_hours * sleep_hours
physical_social_interaction = physical_activity_hours * social_hours
gpa_squared = gpa ** 2

# Binning Study Hours
bins = [0, 2, 4, 6, np.inf]
labels = ['Low', 'Medium', 'High', 'Very High']
study_bin = pd.cut([study_hours], bins=bins, labels=labels)[0]

# One-hot encode Study_Hours_Bin
study_bin_encoded = {
    'Study_Hours_Low': 0,
    'Study_Hours_Medium': 0,
    'Study_Hours_High': 0,
    'Study_Hours_Very High': 0
}
if study_bin is not np.nan:
    study_bin_encoded[f'Study_Hours_{study_bin}'] = 1

# Assemble all features
input_data = pd.DataFrame([{
    'Study_Hours_Per_Day': study_hours,
    'Extracurricular_Hours_Per_Day': extracurricular_hours,
    'Sleep_Hours_Per_Day': sleep_hours,
    'Social_Hours_Per_Day': social_hours,
    'Physical_Activity_Hours_Per_Day': physical_activity_hours,
    'GPA': gpa,
    'Study_Sleep_Interaction': study_sleep_interaction,
    'Physical_Social_Interaction': physical_social_interaction,
    'GPA_squared': gpa_squared,
    **study_bin_encoded
}])

# Prediction
if st.button("Prediksi Tingkat Stres"):
    prediction = model.predict(input_data)[0]
    st.success(f"Tingkat stres yang diprediksi: **{prediction}**")
