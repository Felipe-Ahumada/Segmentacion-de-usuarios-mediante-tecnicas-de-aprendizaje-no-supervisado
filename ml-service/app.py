"""API REST del servicio de segmentación de usuarios.

Expone los resultados del modelo KMeans entrenado por ``train.py`` y permite
clasificar nuevos usuarios en uno de los segmentos existentes. Los modelos se
cargan al iniciar la aplicación desde el volumen ``models/``.
"""

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
    """Verifica que el servicio esté en funcionamiento."""
    return {"mensaje": "Servicio ML funcionando"}


@app.get("/dashboard-data")
def dashboard_data():
    """Retorna usuarios segmentados, centroides y métricas para el dashboard.

    Devuelve ``503`` si los archivos de resultados aún no fueron generados.
    """
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
    """Clasifica un usuario en un segmento a partir de sus variables.

    Recibe un diccionario con las variables del usuario y retorna el cluster
    asignado por el modelo. Devuelve ``503`` si el modelo no está cargado y
    ``400`` si los datos enviados son inválidos.
    """
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecutar train.py primero.")

    # El scaler exige las mismas columnas y en el mismo orden con que se entrenó.
    columnas = list(scaler.feature_names_in_)
    faltantes = [c for c in columnas if c not in datos]
    if faltantes:
        raise HTTPException(status_code=400, detail=f"Faltan variables requeridas: {faltantes}")

    try:
        df = pd.DataFrame([datos])[columnas]
        X = scaler.transform(df)
        cluster = modelo.predict(X)
        return {"cluster": int(cluster[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")
