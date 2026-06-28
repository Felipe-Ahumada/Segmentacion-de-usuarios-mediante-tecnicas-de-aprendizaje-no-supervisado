import json
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Servicio Segmentación Usuarios Streaming")

try:
    modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    with open("models/metricas.json") as f:
        metricas = json.load(f)
    print("Modelos y métricas cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error al cargar modelos: {e}. Ejecutar train.py antes de iniciar la API.")
    modelo = None
    scaler = None
    metricas = {}


@app.get("/")
def inicio():
    return {"mensaje": "Servicio ML funcionando"}


@app.get("/dashboard-data")
def dashboard_data():
    try:
        usuarios = pd.read_csv("outputs/usuarios_segmentados.csv")
        centroides = pd.read_csv("outputs/centroides.csv")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Resultados no disponibles: {e}")
    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas,
    }


@app.post("/predict")
def predict(datos: dict):
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecutar train.py primero.")
    try:
        df = pd.DataFrame([datos])
        X = scaler.transform(df)
        cluster = modelo.predict(X)
        return {"cluster": int(cluster[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")
