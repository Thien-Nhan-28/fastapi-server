from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

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

    # Dự đoán bệnh
    preds = mlp_model.predict(input_df)
    pred_class = np.argmax(preds, axis=1)

    # Giải mã tên bệnh
    pred_disease = disease_encoder.inverse_transform(pred_class)[0]

    # Lấy chuyên khoa tương ứng
    specialty = df_disease_specialty[df_disease_specialty['diseases'] == pred_disease]['specialty'].values
    specialty_name = specialty[0] if len(specialty) > 0 else "Unknown"
    print("Specialty found:", specialty_name)
    print("Disease predicted:", pred_disease)
    print("Specialty found:", specialty_name)
    print("Specialty array:", specialty)

    # Trả về cả bệnh và chuyên khoa
    return {
        "predicted_disease": pred_disease,
        "predicted_specialty": specialty_name
    }
