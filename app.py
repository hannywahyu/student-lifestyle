import streamlit as st

# Judul Dashboard
st.title("🎓 Dashboard Prediksi Tingkat Stres Mahasiswa")

# Deskripsi Aplikasi
st.markdown("""
Selamat datang di **Aplikasi Prediksi Tingkat Stres Mahasiswa**.

Aplikasi ini bertujuan membantu memprediksi tingkat stres mahasiswa berdasarkan aktivitas harian seperti jam belajar, tidur, aktivitas fisik, dan lainnya.

---

### 🧩 Fitur Utama:
- 📊 **Input Data Gaya Hidup**: Masukkan data aktivitas harian.
- 🧠 **Prediksi Stres**: Dapatkan estimasi tingkat stres berdasarkan model machine learning.
- 📈 **Visualisasi**: (Opsional) Lihat grafik interaktif dari distribusi data.

---

### 📝 Petunjuk Penggunaan:
1. Buka tab **Prediksi** untuk mulai menggunakan model.
2. Masukkan informasi sesuai aktivitas Anda sehari-hari.
3. Tekan tombol **Prediksi Tingkat Stres** untuk melihat hasil.

---

# Tambahan estetika opsional
st.image("https://img.freepik.com/free-vector/stress-concept-illustration_114360-10140.jpg", use_column_width=True)
