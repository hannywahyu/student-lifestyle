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

model = load_model()

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

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Identitas", "Data Description", "Prediction", "About"])

# ===================== Halaman Identitas =====================
if page == "Identitas":
    st.title("👤 Halaman Identitas Pengguna")

    # Input nama dan umur
    nama = st.text_input("Masukkan Nama Anda:")
    umur = st.number_input("Masukkan Umur Anda:", min_value=0, max_value=120, value=20)

    if nama.strip():
        st.success(f"Halo **{nama}**, umur Anda **{umur} tahun**")
        # Simpan di session state agar bisa digunakan di halaman lain
        st.session_state["nama"] = nama
        st.session_state["umur"] = umur
    else:
        st.warning("Silakan masukkan nama terlebih dahulu.")

# ===================== Halaman Data Description =====================
elif page == "Data Description":
    st.title("📊 Deskripsi Data")
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
    st.subheader("Cuplikan Data")
    st.dataframe(data)

# ===================== Halaman Prediction =====================
elif page == "Prediction":
    st.title("📈 Prediksi Tingkat Stres")

    if "nama" in st.session_state and "umur" in st.session_state:
        st.info(f"Prediksi untuk: **{st.session_state['nama']}**, umur **{st.session_state['umur']} tahun**")
    else:
        st.warning("Silakan isi identitas terlebih dahulu di halaman *Identitas*.")

    st.write("Masukkan informasi berikut untuk memprediksi tingkat stres:")

    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    sleep_duration = st.slider("Sleep Duration per Day (hours)", 0, 12, 7)
    physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 3)
    social_hours = st.slider("Social Hours per Day", 0, 12, 2)
    extracurricular = st.selectbox("Ikut Kegiatan Ekstrakurikuler?", ["Yes", "No"])
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)

    extracurricular_binary = 1 if extracurricular == "Yes" else 0

    input_data = pd.DataFrame([{
    "Study Hours": study_hours,
    "Sleep Duration": sleep_duration,
    "Physical Activity": physical_activity,
    "Social Hours": social_hours,
    "Extracurricular Activities": extracurricular_binary,
    "GPA": gpa,
    "Level": 0  # sesuaikan jika memang ada fitur ini
}])

    if st.button("Prediksi"):
        prediction = model.predict(input_data)[0]
        if "nama" in st.session_state:
            st.success(f"{st.session_state['nama']}, tingkat stres kamu diprediksi: **{prediction}**")
        else:
            st.success(f"Tingkat stres diprediksi: **{prediction}**")

# ===================== Halaman About =====================
elif page == "About":
    st.title("ℹ️ Tentang Model Ini")
    st.write("""
    Model ini menggunakan pendekatan **Stacking Classifier** untuk memprediksi tingkat stres mahasiswa. 
    Stacking adalah metode ensemble machine learning yang menggabungkan beberapa model dasar dan meta untuk meningkatkan akurasi.

    - **Model Base**: Kombinasi dari beberapa algoritma
    - **Model Meta**: Menggabungkan output dari model base
    - **Kelebihan**: Meningkatkan akurasi dan generalisasi
    """)
