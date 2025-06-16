import streamlit as st
st.set_page_config(page_title="Deteksi Kanker Payudara", page_icon="🩺", layout="wide")

import numpy as np
import joblib
from PIL import Image
import onnxruntime as ort

# === Load model sekali saja ===
@st.cache_resource
def load_models():
    resnet = ort.InferenceSession("resnet50_feature_extractor.onnx")
    lgb = joblib.load("lightgbm_classifier_optimized.pkl")
    return resnet, lgb

resnet_model, lgb_model = load_models()
class_labels = {0: "Benign", 1: "Malignant", 2: "Normal"}

# === SIDEBAR ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
st.sidebar.markdown("### 🧬replikasi Deteksi Kanker Payudara")
st.sidebar.markdown("**Mata Kuliah:** Kecerdasan Buatan  \n**Kelompok:** 8")
st.sidebar.info(
    "🔍 Aplikasi ini menggunakan CNN (ResNet50 via ONNX) untuk ekstraksi fitur dari gambar mamografi, "
    "kemudian diklasifikasi menggunakan LightGBM."
)

# === HEADER UTAMA ===
st.markdown("<h1 style='text-align: center;'>📷 Sistem Deteksi Kanker Payudara Otomatis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah gambar mamografi untuk deteksi dini kanker: <b>Benign</b>, <b>Malignant</b>, atau <b>Normal</b>.</p>", unsafe_allow_html=True)

st.divider()

# === FORM PASIEN ===
with st.expander("🗾 Formulir Data Pasien"):
    nama = st.text_input("👤 Nama Lengkap")
    usia = st.number_input("🎂 Usia", min_value=1, max_value=120, value=30)
    tanggal = st.date_input("🗕 Tanggal Pemeriksaan")

# === UPLOAD GAMBAR ===
uploaded_file = st.file_uploader("📄 Upload Gambar Mamografi", type=["jpg", "jpeg", "png"])

# === PROSES DAN KLASIFIKASI ===
if uploaded_file:
    try:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🖼️ Gambar Mamografi", use_column_width=True)

        with col2:
            st.info("🔎 Gambar sedang dianalisis...")

            # Preprocessing gambar
            image = image.resize((224, 224))
            img_array = np.array(image).astype(np.float32)
            img_array = img_array / 255.0  # Normalisasi sederhana [0, 1]
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi dengan ONNX
            inputs = {resnet_model.get_inputs()[0].name: img_array}
            features = resnet_model.run(None, inputs)[0]
            prediction = lgb_model.predict(features)
            result_index = int(prediction[0])
            result = class_labels.get(result_index, "Unknown")

            # Tampilkan hasil
            st.subheader("🧠 Hasil Klasifikasi")
            if result == "Benign":
                st.success("🟢 Benign (Jinak)")
                st.markdown("Tumor jinak cenderung tidak menyebar. Disarankan tetap memantau secara berkala.")
            elif result == "Malignant":
                st.error("🔴 Malignant (Ganas)")
                st.markdown("Tumor ganas bersifat agresif. Segera lakukan pemeriksaan lanjutan ke dokter.")
            elif result == "Normal":
                st.success("✅ Normal")
                st.markdown("Tidak ditemukan indikasi kelainan. Pemeriksaan rutin tetap diperlukan.")

            # === Confidence Score ===
            if st.checkbox("📊 Tampilkan Confidence Score (%)", value=True):
                if hasattr(lgb_model, "predict_proba"):
                    proba = lgb_model.predict_proba(features)[0]
                    persentase = np.round(proba * 100, 2)

                    st.markdown("#### 📊 Confidence Score")
                    for label, score in zip(class_labels.values(), persentase):
                        emoji = "🟢" if label == result else "⚪"
                        st.markdown(f"{emoji} **{label}**: {score:.2f}%")
                        st.progress(float(score) / 100)

                    st.markdown("#### 📜 Ringkasan Confidence Score")
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
    - **Benign**: Tumor tidak ganas dan tidak menyebar ke jaringan lain.
    - **Malignant**: Tumor ganas yang berpotensi menyebar dan memerlukan penanganan segera.
    - **Normal**: Tidak ditemukan kelainan pada hasil mamografi.

    👉 Tetap lakukan deteksi dini dan konsultasi dengan tenaga medis profesional secara berkala.
    """)
