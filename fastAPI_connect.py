import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

from fastapi import FastAPI, HTTPException, File, UploadFile
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

fe = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
skin_model = joblib.load("models/skin_disease.joblib")
diabetes_model = tf.keras.models.load_model("models/diabetes_model.keras")
heart_disease_model = tf.keras.models.load_model("models/heart_disease_model.keras")
diabetes_scaler = joblib.load("models/db_scaler.joblib")
heart_scaler = joblib.load("models/hd_scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")


def skd_run(contents1):
    try:
        image = Image.open(io.BytesIO(contents1)).convert("RGB").resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error in pic loading")
        return None

img_p = "pre_start_test_img/ring.jpg"

if os.path.exists(img_p):
    with open(img_p, "rb") as img_f:
        img_b = img_f.read()

img_array = skd_run(img_b)
feature_test = fe.predict( img_array)
prediction = skin_model.predict(feature_test)

def db_run(inp_data):
            inp_arr = np.array([list(inp_data.dict().values())])
            inp_scaled = diabetes_scaler.transform(inp_arr)
            return  inp_scaled
test_db = db_run(DiabetesInput(glucose=181, BP=64, insulin=180, BMI=35, age=36))
test_res = diabetes_model.predict(test_db)[0][0]

def hd_run(inp_data):
            inp_arr = np.array([list(inp_data.dict().values())])
            inp_scaled = heart_scaler.transform(inp_arr)
            return inp_scaled
test_hd = hd_run(HeartDiseaseInput(age=38, gender=1, cp=2, trestbps=138, chol=175, restecg=1, thalach=173, exang=0, oldpeak=0, slope=2, ca=4))
test_res1= heart_disease_model.predict(test_hd)[0][0]

@app.post("/predict/skin_disease")
async def predict_skin_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = skd_run(contents)
        feature = fe.predict( img_array)
        prediction = skin_model.predict(feature)
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        return JSONResponse(content={"result": predicted_disease.tolist()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict/diabetes")
async def predict_diabetes(input_data: DiabetesInput):
    try:
        input_scaled = db_run(input_data)
        result = diabetes_model.predict(input_scaled)[0][0]
        if result > 0.5:
            diagnosis = "You have diabetes"
        else:
            diagnosis = "You don't have diabetes"
        return {"result": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/heart_disease")
async def predict_heart_disease(input_data: HeartDiseaseInput):
    try:
        input_scaled= hd_run(input_data)
        result = heart_disease_model.predict(input_scaled)[0][0]
        if result > 0.5:
            diagnosis = "You have Heart disease"  
        else:
            diagnosis = "You don't have Heart disease"
        return {"result": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)