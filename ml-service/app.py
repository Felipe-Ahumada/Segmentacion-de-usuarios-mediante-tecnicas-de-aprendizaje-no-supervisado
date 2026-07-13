"""API REST que expone el modelo de segmentación y clasifica nuevos usuarios.

Carga el modelo KMeans y el scaler entrenados por train.py desde el volumen
models/ al iniciar, y sirve los resultados al dashboard.
"""

import json
import logging
import pickle

import numpy as np
import preprocesamiento  # noqa: F401 — pickle necesita el módulo para deserializar Winsorizer/CorrelationFilter
import pandas as pd
from fastapi import FastAPI, HTTPException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Servicio Segmentación Usuarios Streaming")

def _forzar_float64(obj):
    """Convierte arrays internos de un estimador sklearn a float64."""
    for attr in ("mean_", "scale_", "var_", "cluster_centers_"):
        if hasattr(obj, attr):
            setattr(obj, attr, getattr(obj, attr).astype(np.float64))

try:
    modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    _forzar_float64(scaler)
    _forzar_float64(modelo)
    with open("models/metricas.json") as f:
        metricas = json.load(f)
    logger.info("Modelos y métricas cargados correctamente.")
except FileNotFoundError as e:
    logger.error("Error al cargar modelos: %s. Ejecutar train.py antes de iniciar la API.", e)
    modelo = None
    scaler = None
    metricas = {}

try:
    clasificador = pickle.load(open("models/mejor_clasificador.pkl", "rb"))
    regresor = pickle.load(open("models/mejor_regresor.pkl", "rb"))
    with open("models/metricas_supervisado.json") as f:
        metricas_supervisado = json.load(f)
    logger.info("Modelos supervisados cargados correctamente.")
except FileNotFoundError:
    clasificador = None
    regresor = None
    metricas_supervisado = {}

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


@app.get("/supervised-data")
def supervised_data():
    """Retorna métricas y resultados de los modelos supervisados."""
    if not metricas_supervisado:
        raise HTTPException(status_code=503, detail="Métricas supervisadas no disponibles.")
    return metricas_supervisado


@app.post("/predict-clasificador")
def predict_clasificador(datos: dict):
    """Clasifica un usuario usando el mejor clasificador supervisado."""
    if clasificador is None or scaler is None:
        raise HTTPException(status_code=503, detail="Clasificador no disponible.")

    columnas = list(scaler.feature_names_in_)
    faltantes = [c for c in columnas if c not in datos]
    if faltantes:
        raise HTTPException(status_code=400, detail=f"Faltan variables requeridas: {faltantes}")

    try:
        df = pd.DataFrame([datos])[columnas].astype(float)
        for c in FRACCIONES:
            df[c] = df[c] / 100
        cluster = clasificador.predict(df)
        return {"cluster": int(cluster[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")


@app.post("/predict-gasto")
def predict_gasto(datos: dict):
    """Predice el gasto mensual de un usuario usando el mejor regresor."""
    if regresor is None:
        raise HTTPException(status_code=503, detail="Regresor no disponible.")

    features_reg = metricas_supervisado.get("regresion", {}).get("features", [])
    faltantes = [c for c in features_reg if c not in datos]
    if faltantes:
        raise HTTPException(status_code=400, detail=f"Faltan variables requeridas: {faltantes}")

    try:
        df = pd.DataFrame([datos])[features_reg].astype(float)
        for c in FRACCIONES:
            if c in df.columns:
                df[c] = df[c] / 100
        pred = regresor.predict(df)
        return {"gasto_mensual_estimado": round(float(pred[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")


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
