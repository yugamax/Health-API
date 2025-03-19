import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import io
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_models():
    global fe, skin_model, diabetes_model, heart_disease_model, diabetes_scaler, heart_scaler, label_encoder
    fe = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    skin_model = joblib.load("models/skin_disease.joblib")
    diabetes_model = tf.keras.models.load_model("models/diabetes_model.keras")
    heart_disease_model = tf.keras.models.load_model("models/heart_disease_model.keras")
    diabetes_scaler = joblib.load("models/db_scaler.joblib")
    heart_scaler = joblib.load("models/hd_scaler.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")

load_models()

class DiabetesInput(BaseModel):
    glucose: float
    BP: float
    insulin: float
    BMI: float
    age: float

class HeartDiseaseInput(BaseModel):
    age: int
    gender: int
    cp: int
    trestbps: int
    chol: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int

def process_image(contents: bytes):
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def process_diabetes(input_data: DiabetesInput):
    inp_arr = np.array([list(input_data.dict().values())])
    return diabetes_scaler.transform(inp_arr)

def process_heart_disease(input_data: HeartDiseaseInput):
    inp_arr = np.array([list(input_data.dict().values())])
    return heart_scaler.transform(inp_arr)

sk_router = APIRouter()
disease_router = APIRouter()

@sk_router.post("/predict/skin_disease")
async def predict_skin_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = process_image(contents)
        features = fe.predict(img_array)
        prediction = skin_model.predict(features)
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        return JSONResponse(content={"result": predicted_disease.tolist()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@disease_router.post("/predict/diabetes")
async def predict_diabetes(input_data: DiabetesInput):
    try:
        input_scaled = process_diabetes(input_data)
        result = diabetes_model.predict(input_scaled)[0][0]
        diagnosis = "You have diabetes" if result > 0.5 else "You don't have diabetes"
        return {"result": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@disease_router.post("/predict/heart_disease")
async def predict_heart_disease(input_data: HeartDiseaseInput):
    try:
        input_scaled = process_heart_disease(input_data)
        result = heart_disease_model.predict(input_scaled)[0][0]
        diagnosis = "You have Heart disease" if result > 0.5 else "You don't have Heart disease"
        return {"result": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(sk_router, prefix="/image")
app.include_router(disease_router, prefix="/disease")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
