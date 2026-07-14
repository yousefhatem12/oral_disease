from fastapi import FastAPI, File, UploadFile, HTTPException

# 1. استيراد الـ CORS Middleware
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

app = FastAPI(
    title="Oral Diseases Classification API (ResNet50)",
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # يسمح لجميع الواجهات بالاتصال
    allow_credentials=True,
    allow_methods=["*"], # يسمح بجميع الدوال (GET, POST...)
    allow_headers=["*"], # يسمح بجميع الـ Headers
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "resnet_model.h5")

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("--- Model Loaded Successfully from outside backend folder ---")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    MODEL = None

CLASS_NAMES = ["benign_lesions", "malignant_lesions"]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.resnet50.preprocess_input(img_array)
    return processed_img


@app.get("/")
async def root():
    return {"status": "online", "message": "Backend API is running professionally!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        predictions = MODEL.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "raw_predictions": {
                CLASS_NAMES[0]: float(predictions[0][0]),
                CLASS_NAMES[1]: float(predictions[0][1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)