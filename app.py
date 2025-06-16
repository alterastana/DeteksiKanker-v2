import streamlit as st
st.set_page_config(page_title="Deteksi Kanker Payudara", page_icon="🩺", layout="wide")

import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load model dengan caching
@st.cache_resource
def load_models():
    resnet = load_model("resnet50_feature_extractor.keras")
    lgb = joblib.load("lightgbm_classifier_optimized.pkl")
    return resnet, lgb

resnet_model, lgb_model = load_models()
class_labels = {0: "Benign", 1: "Malignant", 2: "Normal"}

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
st.sidebar.markdown("### 🧬 Aplikasi Deteksi Kanker Payudara")
st.sidebar.markdown("**Mata Kuliah: Kecerdasan Buatan**  \n**Kelompok 8**")
st.sidebar.info(
    "🔍 Model CNN (ResNet50) digunakan untuk ekstraksi fitur dari gambar mamografi, "
    "kemudian diklasifikasikan menggunakan LightGBM. Optimasi dilakukan dengan algoritma "
    "**Root Mean Square Propagation (RMSProp)**."
)

# Header
st.markdown("<h1 style='text-align: center;'>📷 Sistem Deteksi Otomatis Kanker Payudara</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mamografi untuk mengklasifikasi: <b>Benign</b>, <b>Malignant</b>, atau <b>Normal</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

# Formulir Pasien
with st.expander("🧾 Formulir Pasien"):
    nama = st.text_input("👤 Nama Pasien")
    usia = st.number_input("🎂 Usia", min_value=1, max_value=120, value=30)
    tanggal = st.date_input("📅 Tanggal Pemeriksaan")

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload Gambar Mamografi", type=["jpg", "jpeg", "png"])

# Prediksi jika gambar ada
if uploaded_file:
    try:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🖼️ Gambar Mamografi", use_column_width=True)

        with col2:
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
                st.markdown("Tumor jinak umumnya tidak menyebar dan tidak bersifat agresif. Tetap lakukan pemeriksaan berkala.")
            elif result == "Malignant":
                st.error("🔴 Hasil: Malignant (Ganas)")
                st.markdown("Tumor ganas dapat menyebar cepat. Segera konsultasikan ke dokter spesialis.")
            elif result == "Normal":
                st.success("✅ Hasil: Normal")
                st.markdown("Tidak ditemukan indikasi kelainan. Pemeriksaan rutin tetap disarankan.")

            # Confidence Score
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
                    st.warning("⚠️ Model tidak mendukung probabilitas prediksi.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
else:
    st.warning("👈 Silakan unggah gambar terlebih dahulu.")

# Edukasi
with st.expander("ℹ️ Tentang Kanker Payudara"):
    st.markdown("""
    - **Benign**: Tumor tidak ganas, tidak menyebar. Tetap perlu pemantauan.
    - **Malignant**: Kanker ganas. Butuh penanganan medis segera.
    - **Normal**: Tidak ada indikasi kelainan.

    👉 Lakukan pemeriksaan rutin dan konsultasikan dengan tenaga medis profesional.
    """)
