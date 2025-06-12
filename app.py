from flask import Flask, request, render_template
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# === Flask App Setup ===
app = Flask(__name__)

# === Load Models ===
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
xgb_model = joblib.load("xgboost_model.pkl")

# Label klasifikasi sesuai indeks model
class_labels = {0: "Benign", 1: "Malignant", 2: "Normal"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    result = None
    error = None

    if 'file' not in request.files:
        error = "No file part in the request."
    else:
        file = request.files['file']
        if file.filename == '':
            error = "No file selected for uploading."
        else:
            try:
                image = Image.open(file.stream).convert("RGB").resize((224, 224))
                img_array = img_to_array(image)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                features = resnet_model.predict(img_array, verbose=0)
                prediction = xgb_model.predict(features)
                result = class_labels.get(int(prediction[0]), "Unknown")
            except Exception as e:
                error = f"Prediction failed: {str(e)}"

    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
