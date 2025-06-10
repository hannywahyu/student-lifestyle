import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
#with open('best_stress_prediction_model.pkl', 'rb') as file:
#model = pickle.load(file)

# Judul aplikasi
st.title('Prediksi Tingkat Stres Mahasiswa')

st.write('Masukkan data aktivitas harian Anda untuk memprediksi tingkat stres.')

# Input user
study = st.number_input('Jam Belajar per Hari', min_value=0.0, max_value=24.0, step=0.5)
extracurricular = st.number_input('Jam Ekstrakurikuler per Hari', min_value=0.0, max_value=24.0, step=0.5)
sleep = st.number_input('Jam Tidur per Hari', min_value=0.0, max_value=24.0, step=0.5)
social = st.number_input('Jam Sosialisasi per Hari', min_value=0.0, max_value=24.0, step=0.5)
physical = st.number_input('Jam Aktivitas Fisik per Hari', min_value=0.0, max_value=24.0, step=0.5)
gpa = st.number_input('IPK', min_value=0.0, max_value=4.0, step=0.01)

# Prediksi ketika tombol ditekan
if st.button('Prediksi Tingkat Stres'):
    # Fitur interaksi
    study_sleep_interaction = study * sleep
    physical_social_interaction = physical * social
    gpa_squared = gpa ** 2

    # Binning jam belajar
    bins = [0, 2, 4, 6, np.inf]
    labels = ['Low', 'Medium', 'High', 'Very High']
    study_bin = pd.cut([study], bins=bins, labels=labels)[0]

    # One-hot encoding manual
    study_bin_encoded = {
        'Study_Hours_Low': 0,
        'Study_Hours_Medium': 0,
        'Study_Hours_High': 0,
        'Study_Hours_Very High': 0
    }
    if study_bin is not pd.NA:
        study_bin_encoded[f'Study_Hours_{study_bin}'] = 1

    # Susun fitur sesuai urutan pelatihan
    features = [
        study,
        extracurricular,
        sleep,
        social,
        physical,
        gpa,
        study_sleep_interaction,
        physical_social_interaction,
        gpa_squared,
        study_bin_encoded['Study_Hours_Low'],
        study_bin_encoded['Study_Hours_Medium'],
        study_bin_encoded['Study_Hours_High'],
        study_bin_encoded['Study_Hours_Very High']
    ]

    # Prediksi
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    # Tampilkan hasil
    st.success(f'Prediksi Tingkat Stres: **{prediction}**')

