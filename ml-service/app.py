"""API REST que expone el modelo de segmentación y clasifica nuevos usuarios.

Carga el modelo KMeans y el scaler entrenados por train.py desde el volumen
models/ al iniciar, y sirve los resultados al dashboard.
"""

import json
import logging
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Servicio Segmentación Usuarios Streaming")

try:
    modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    with open("models/metricas.json") as f:
        metricas = json.load(f)
    logger.info("Modelos y métricas cargados correctamente.")
except FileNotFoundError as e:
    logger.error("Error al cargar modelos: %s. Ejecutar train.py antes de iniciar la API.", e)
    modelo = None
    scaler = None
    metricas = {}

# Los porcentajes se reciben en escala 0-100. Las fracciones se convierten a 0-1
# (escala con que se entrenó el modelo) justo antes de predecir.
FRACCIONES = ("porcentaje_uso_promociones", "porcentaje_uso_app_movil")


@app.get("/")
def inicio():
    """Verifica que el servicio esté en funcionamiento."""
    return {"mensaje": "Servicio ML funcionando"}


@app.get("/dashboard-data")
def dashboard_data():
    """Retorna usuarios segmentados, centroides y métricas para el dashboard.

    Responde 503 si train.py todavía no generó los archivos de outputs/.
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
    """Clasifica un usuario en un segmento con el modelo ya entrenado.

    Todos los porcentajes se reciben en escala 0-100. Responde 503 si el modelo
    no está cargado y 400 si faltan variables o los datos enviados son inválidos.
    """
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecutar train.py primero.")

    # El scaler exige las mismas columnas y en el mismo orden con que se entrenó.
    columnas = list(scaler.feature_names_in_)
    faltantes = [c for c in columnas if c not in datos]
    if faltantes:
        raise HTTPException(status_code=400, detail=f"Faltan variables requeridas: {faltantes}")

    try:
        df = pd.DataFrame([datos])[columnas].astype(float)
        # Las fracciones llegan en 0-100 y el modelo las espera en 0-1.
        for c in FRACCIONES:
            df[c] = df[c] / 100
        X = scaler.transform(df)
        cluster = modelo.predict(X)
        return {"cluster": int(cluster[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")
