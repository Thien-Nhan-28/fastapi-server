from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 💡 Thêm dòng này
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Cho phép truy cập từ frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🎯 Thêm origin React của bạn ở đây
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model MLP
mlp_model = tf.keras.models.load_model("model/mlp_model.keras")

# Load encoder, symptom columns và bảng disease-specialty
disease_encoder = joblib.load('model/disease_encoder.pkl')
symptom_columns = joblib.load('model/symptom_columns.pkl')
df_disease_specialty = pd.read_pickle('model/disease_specialty.pkl')  # chứa cột diseases và specialty

class SymptomsRequest(BaseModel):
    symptoms: List[str]

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/symptoms")
def get_symptoms():
    return {"symptoms": symptom_columns}

@app.get("/disease_labels")
def get_disease_labels():
    labels = disease_encoder.classes_.tolist()
    return {"disease_labels": labels}

@app.get("/couple")
def get_couple():
    data = df_disease_specialty.to_dict(orient='records')
    return {"df_disease_specialty": data}

@app.post("/predict")
def predict_disease(request: SymptomsRequest):
    input_symptoms = request.symptoms

    # Tạo vector đặc trưng 0/1
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptom_columns]
    input_df = pd.DataFrame([input_vector], columns=symptom_columns)

    # Dự đoán xác suất các bệnh
    preds = mlp_model.predict(input_df)  # preds là mảng xác suất shape (1, num_classes)

    # Lấy 2 chỉ số có xác suất cao nhất
    top_indices = preds[0].argsort()[-3:][::-1]

    # Tạo list chứa tên bệnh và chuyên khoa tương ứng
    predicted_diseases = []
    predicted_specialties = []

    for idx in top_indices:
        disease_name = disease_encoder.inverse_transform([idx])[0]
        specialty = df_disease_specialty[df_disease_specialty['diseases'] == disease_name]['specialty'].values
        specialty_name = specialty[0] if len(specialty) > 0 else "Unknown"

        predicted_diseases.append(disease_name)
        predicted_specialties.append(specialty_name)

    # Debug log
    print("Predicted diseases:", predicted_diseases)
    print("Predicted specialties:", predicted_specialties)

    return {
        "predicted_disease": predicted_diseases,
        "predicted_specialty": predicted_specialties
    }
