import os
import uuid
import joblib
import requests
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image


model_path = "knn_banana_disease.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Trained KNN model not found! Please run training script first.")

print("📥 Loading saved KNN model...")
knn = joblib.load(model_path)

# MobileNetV2 (ayaw saba hehehe)
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=False, pooling="avg"
)


class_labels = [
    " Banana Black Sigatoka Disease",
    " Banana Bract Mosaic Virus Disease",
    " Banana Healthy Leaf",
    " Banana Insect Pest Disease",
    " Banana Moko Disease",
    " Banana Panama Disease",
    " Banana Yellow Sigatoka Disease",
]


app = FastAPI(title="Banana Disease Detection API")


DOWNLOAD_DIR = os.path.join(os.getcwd(), "Download")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def preprocess_image(img_path):
    """Load and preprocess an image for MobileNetV2 feature extraction."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(img_path):
    """Predict disease from image path."""
    img_array = preprocess_image(img_path)
    features = base_model.predict(img_array)
    pred_class = knn.predict(features)[0]
    probs = knn.predict_proba(features)[0]
    confidence = np.max(probs) * 100
    return class_labels[pred_class], confidence, probs


@app.get("/predict")
def predict(image_url: str = Query(..., description="Public URL of the image to scan")):
    try:
        # Add headers to bypass restrictions (e.g. Imgur)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, stream=True, headers=headers, timeout=20)

        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to download image. Status {response.status_code}"},
            )

       
        temp_filename = os.path.join(DOWNLOAD_DIR, f"banana_{uuid.uuid4().hex}.jpg")
        with open(temp_filename, "wb") as f:
            f.write(response.content)

       
        disease, conf, probs = predict_disease(temp_filename)

    
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        return {
            "predicted_disease": disease.strip(),
            "confidence": f"{conf:.2f}%",
            "class_probabilities": {
                label.strip(): f"{p*100:.2f}%" for label, p in zip(class_labels, probs)
            },
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

