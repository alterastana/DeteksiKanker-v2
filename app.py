from flask import Flask, request, render_template, url_for
import numpy as np
import joblib
import os
import uuid
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# === Inisialisasi Aplikasi Flask ===
app = Flask(__name__)

# === Konfigurasi Folder Upload ===
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Muat Model ===
resnet_model = load_model("resnet50_feature_extractor.keras")
lgb_model = joblib.load("lightgbm_classifier_optimized.pkl")

# Label klasifikasi berdasarkan indeks prediksi model
class_labels = {0: "Benign", 1: "Malignant", 2: "Normal"}

# Halaman utama (form upload)
@app.route('/')
def home():
    return render_template('index.html', result=None, error=None, filename=None)

# Endpoint untuk menangani form submission (POST)
@app.route('/', methods=['POST'])
def predict():
    result = None
    error = None
    filename = None

    # Periksa apakah ada file dalam request
    if 'file' not in request.files:
        error = "Tidak ada file dalam permintaan."
    else:
        file = request.files['file']
        # Periksa apakah file kosong
        if file.filename == '':
            error = "Tidak ada file yang dipilih untuk diunggah."
        else:
            try:
                # Hapus semua file lama di folder uploads
                for f in os.listdir(app.config['UPLOAD_FOLDER']):
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

                # Simpan file dengan nama unik
                original_filename = secure_filename(file.filename)
                ext = os.path.splitext(original_filename)[1]
                filename = f"{uuid.uuid4().hex}{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Log lokasi penyimpanan file
                print("File disimpan di:", filepath)

                # Proses gambar
                image = Image.open(filepath).convert("RGB").resize((224, 224))
                img_array = img_to_array(image)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Ekstraksi fitur dan lakukan prediksi
                features = resnet_model.predict(img_array, verbose=0)
                prediction = lgb_model.predict(features)
                result = class_labels.get(int(prediction[0]), "Unknown")

            except Exception as e:
                error = f"Prediksi gagal: {str(e)}"
                filename = None
                print("[ERROR]", error)

    return render_template('index.html', result=result, error=error, filename=filename)

# Jalankan aplikasi Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

