import streamlit as st
st.set_page_config(page_title="Deteksi Kanker Payudara", page_icon="🩺", layout="wide")

import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_lottie import st_lottie
import requests

# === Fungsi: Memuat animasi dari URL ===
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# === Animasi ===
lottie_cancer = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_checkup = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_jk6c1n2h.json")

# === Load model (cached) ===
@st.cache_resource
def load_models():
    resnet = load_model("resnet50_feature_extractor.keras")
    lgb = joblib.load("lightgbm_classifier_optimized.pkl")
    return resnet, lgb

resnet_model, lgb_model = load_models()
class_labels = {0: "Benign", 1: "Malignant", 2: "Normal"}

# === SIDEBAR ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
st.sidebar.markdown("### 🧬 Aplikasi Deteksi Kanker Payudara")
st.sidebar.markdown("**Mata Kuliah: Kecerdasan Buatan**  \n**Kelompok 8**")
st.sidebar.info(
    "🔍 Aplikasi ini menggunakan model CNN (ResNet50) untuk mengekstrak fitur dari gambar mamografi, "
    "kemudian mengklasifikasikannya menggunakan LightGBM. Proses pelatihan dioptimasi dengan algoritma "
    "**Root Mean Square Propagation (RMSProp)** untuk meningkatkan akurasi deteksi."
)

# === HEADER ===
st.markdown("<h1 style='text-align: center;'>📷 Sistem Deteksi Otomatis Kanker Payudara</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mamografi untuk mendeteksi kategori kanker: <b>Benign</b>, <b>Malignant</b>, atau <b>Normal</b>.</p>", unsafe_allow_html=True)

if lottie_cancer:
    st_lottie(lottie_cancer, height=200, key="header")

st.markdown("---")

# === FORM PASIEN ===
with st.expander("🧾 Formulir Pasien"):
    nama = st.text_input("👤 Nama Pasien")
    usia = st.number_input("🎂 Usia", min_value=1, max_value=120, value=30)
    tanggal = st.date_input("📅 Tanggal Pemeriksaan")

# === UPLOAD GAMBAR ===
uploaded_file = st.file_uploader("📤 Upload Gambar Mamografi", type=["jpg", "jpeg", "png"])

# === PROSES DAN KLASIFIKASI ===
if uploaded_file:
    try:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🖼️ Gambar Mamografi", use_column_width=True)

        with col2:
            if lottie_checkup:
                st_lottie(lottie_checkup, height=150, key="loading")
            st.info("🔎 Gambar sedang diproses...")

            image = image.resize((224, 224))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            features = resnet_model.predict(img_array, verbose=0)
            prediction = lgb_model.predict(features)
            result_index = int(prediction[0])
            result = class_labels.get(result_index, "Unknown")

            st.subheader("🧠 Hasil Klasifikasi")

            if result == "Benign":
                st.success("🟢 Hasil: Benign (Jinak)")
                st.markdown("Tumor jinak umumnya tidak menyebar dan tidak bersifat agresif. Namun tetap perlu pemantauan medis secara berkala.")
            elif result == "Malignant":
                st.error("🔴 Hasil: Malignant (Ganas)")
                st.markdown("Jenis kanker ganas bersifat agresif dan dapat menyebar cepat. Segera konsultasikan ke dokter spesialis.")
            elif result == "Normal":
                st.success("✅ Hasil: Normal")
                st.markdown("Tidak ditemukan indikasi kelainan mencurigakan. Tetap disarankan melakukan pemeriksaan rutin.")

            # === Confidence Score ===
            if st.checkbox("📈 Tampilkan Confidence Score (%)", value=True):
                if hasattr(lgb_model, "predict_proba"):
                    proba = lgb_model.predict_proba(features)[0]
                    persentase = np.round(proba * 100, 2)

                    st.markdown("#### 🔬 Probabilitas Klasifikasi")
                    for label, score in zip(class_labels.values(), persentase):
                        emoji = "🟢" if label == result else "⚪"
                        st.markdown(f"{emoji} **{label}**: {score:.2f}%")
                        st.progress(float(score) / 100)

                    st.markdown("#### 📋 Tabel Confidence Score")
                    st.table({
                        "Kelas": list(class_labels.values()),
                        "Probabilitas (%)": [f"{p:.2f}%" for p in persentase]
                    })
                else:
                    st.warning("⚠️ Model tidak mendukung prediksi probabilitas.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
else:
    st.warning("👈 Silakan unggah gambar untuk memulai analisis.")

# === EDUKASI TAMBAHAN ===
with st.expander("ℹ️ Tentang Kanker Payudara"):
    st.markdown("""
    - **Benign**: Tumor tidak ganas dan tidak menyebar ke jaringan lain. Umumnya mudah ditangani.
    - **Malignant**: Tumor ganas yang dapat menyebar dan bersifat berbahaya jika tidak ditangani dini.
    - **Normal**: Tidak ditemukan indikasi massa mencurigakan pada gambar mamografi.

    👉 Tetap lakukan pemeriksaan rutin dan konsultasi dengan tenaga medis berlisensi.
    """)
