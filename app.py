import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

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

# Fungsi untuk plotting Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    return fig

# Fungsi untuk plotting ROC Curve
def plot_roc_curve(y_true, y_score, classes):
    y_test_bin = label_binarize(y_true, classes=classes)
    n_classes = y_test_bin.shape[1]

    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    ax.plot([0,1], [0,1], 'k--', lw=2)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    return fig

# Fungsi untuk plotting Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_score, classes):
    y_test_bin = label_binarize(y_true, classes=classes)
    n_classes = y_test_bin.shape[1]

    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f'Class {classes[i]} (AUC = {pr_auc:.2f})')

    ax.set_xlim([0,1])
    ax.set_ylim([0,1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    return fig

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Identitas", "Data Description", "Prediction", "Evaluation", "About"])

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
    sleep_durat_
