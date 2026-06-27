import json
import pickle

import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Servicio Segmentación Usuarios Streaming")

modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

with open("models/metricas.json") as f:
    metricas = json.load(f)


@app.get("/")
def inicio():
    return {"mensaje": "Servicio ML funcionando"}


@app.get("/dashboard-data")
def dashboard_data():
    usuarios = pd.read_csv("data/usuarios_segmentados.csv")
    centroides = pd.read_csv("data/centroides.csv")
    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas,
    }


@app.post("/predict")
def predict(datos: dict):
    df = pd.DataFrame([datos])
    X = scaler.transform(df)
    cluster = modelo.predict(X)
    return {"cluster": int(cluster[0])}
