# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Fungsi untuk load model
@st.cache_resource
def load_model():
    with open("stacking_classifier_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()  # Panggil model di awal agar bisa digunakan di bagian Prediction

# Fungsi untuk load data dummy
@st.cache_data
def load_data(n=300):
    np.random.seed(42)
    data = {
        "Study Hours": np.random.randint(0, 10, size=n),
        "Sleep Duration": np.random.randint(4, 10, size=n),
        "Physical Activity": np.random.randint(0, 5, size=n),
        "Social Hours": np.random.randint(0, 6, size=n),
        "Extracurricular Activities": np.random.randint(0, 2, size=n),
        "GPA": np.round(np.random.uniform(2.0, 4.0, size=n), 2),
        "Level": np.random.randint(0, 3, size=n),
    }
    return pd.DataFrame(data)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Description", "Prediction", "About"])

# ===================== Input Nama dan Umur =====================
with st.expander("👤 Isi Identitas Pengguna", expanded=True):
    nama = st.text_input("Nama Anda:")
    umur = st.number_input("Umur Anda:", min_value=0, max_value=120, value=20, step=1)

    if nama.strip() == "":
        st.warning("Silakan masukkan nama Anda.")
    else:
        st.info(f"Hai **{nama}**, umur Anda **{umur} tahun**")

# ===================== Halaman 1: Data Description =====================
if page == "Data Description":
    st.title("📊 Data Description")
    st.write("""
    Dataset ini berisi informasi tentang tingkat stres mahasiswa yang diukur berdasarkan:
    - **Study Hours**
    - **Sleep Duration**
    - **Physical Activity**
    - **Social Hours**
    - **Extracurricular Activities**
    - **GPA**
    
    Target variabel adalah **Stress Level**.
    """)

    data = load_data()

    st.subheader("Data Preview")
    st.dataframe(data)

# ===================== Halaman 2: Prediction =====================
elif page == "Prediction":
    st.title("📈 Stress Level Prediction")

    st.write("Masukkan informasi berikut untuk memprediksi tingkat stres mahasiswa:")

    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    sleep_duration = st.slider("Sleep Duration per Day (hours)", 0, 12, 7)
    physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 3)
    social_hours = st.slider("Social Hours per Day", 0, 12, 2)
    extracurricular = st.selectbox("Participate in Extracurricular Activities?", ["Yes", "No"])
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)

    extracurricular_binary = 1 if extracurricular == "Yes" else 0

    input_data = pd.DataFrame([{
        "Study Hours": study_hours,
        "Sleep Duration": sleep_duration,
        "Physical Activity": physical_activity,
        "Social Hours": social_hours,
        "Extracurricular Activities": extracurricular_binary,
        "GPA": gpa
    }])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"{nama}, tingkat stres kamu diprediksi: **{prediction}**")

# ===================== Halaman 3: About =====================
elif page == "About":
    st.title("ℹ️ About This Model")

    st.write("""
    Model ini menggunakan pendekatan **Stacking Classifier** untuk memprediksi tingkat stres mahasiswa. 
    Stacking adalah metode ensemble machine learning yang menggabungkan beberapa model dasar dan meta untuk meningkatkan performa prediksi.

    - **Model Base**: Kombinasi beberapa algoritma
    - **Model Meta**: Menggabungkan output dari model base
    - **Kelebihan**: Meningkatkan akurasi dan generalisasi prediksi
    """)
