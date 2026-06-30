"""Tests de integración para la API REST del servicio ML."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import sys
import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))


# Fixtures

@pytest.fixture
def modelo_y_scaler():
    """Crea un modelo KMeans y scaler entrenados con datos sintéticos."""
    np.random.seed(42)
    n = 60
    columnas = [
        "horas_consumo_mensual", "gasto_mensual", "cantidad_contenidos_vistos",
        "sesiones_semana", "porcentaje_finalizacion", "tiempo_promedio_sesion_min",
        "cantidad_generos_consumidos", "porcentaje_uso_promociones",
        "antiguedad_cliente_meses", "edad", "dispositivos_registrados",
        "porcentaje_uso_app_movil", "cantidad_perfiles_creados",
        "interacciones_mensuales_soporte", "distancia_promedio_red_km",
    ]
    data = pd.DataFrame(np.random.rand(n, len(columnas)), columns=columnas)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=3, random_state=29, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans, scaler


@pytest.fixture
def metricas_mock():
    """Métricas simuladas del modelo."""
    return {
        "k_optimo": 3,
        "silhouette_score": 0.25,
        "n_usuarios": 60,
        "n_clusters": 3,
        "varianza_pca": 0.45,
        "inertias": [100, 80, 65, 58, 53, 50, 47, 45, 43],
        "silhouettes": [0.20, 0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.15, 0.14],
        "rango_k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }


@pytest.fixture
def client(modelo_y_scaler, metricas_mock, tmp_path):
    """Cliente de test de FastAPI con modelo y archivos mockeados."""
    kmeans, scaler = modelo_y_scaler

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    pickle.dump(kmeans, open(models_dir / "modelo_kmeans.pkl", "wb"))
    pickle.dump(scaler, open(models_dir / "scaler.pkl", "wb"))
    with open(models_dir / "metricas.json", "w") as f:
        json.dump(metricas_mock, f)

    columnas = list(scaler.feature_names_in_)
    usuarios = pd.DataFrame(np.random.rand(10, len(columnas)), columns=columnas)
    usuarios["cluster"] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    usuarios["pc1"] = np.random.rand(10)
    usuarios["pc2"] = np.random.rand(10)
    usuarios.to_csv(outputs_dir / "usuarios_segmentados.csv", index=False)

    centroides = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=columnas,
    )
    centroides.to_csv(outputs_dir / "centroides.csv", index=False)

    with patch.dict(os.environ, {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "test",
    }):
        with patch("builtins.open", wraps=open):
            import importlib
            old_cwd = os.getcwd()
            os.chdir(tmp_path)
            if "app" in sys.modules:
                del sys.modules["app"]
            import app as api_module
            api_module.modelo = kmeans
            api_module.scaler = scaler
            api_module.metricas = metricas_mock
            tc = TestClient(api_module.app)
            os.chdir(old_cwd)
            yield tc


@pytest.fixture
def usuario_valido():
    """Payload válido para el endpoint POST /predict."""
    return {
        "horas_consumo_mensual": 45,
        "gasto_mensual": 120,
        "cantidad_contenidos_vistos": 18,
        "sesiones_semana": 5,
        "porcentaje_finalizacion": 72,
        "tiempo_promedio_sesion_min": 95,
        "cantidad_generos_consumidos": 6,
        "porcentaje_uso_promociones": 30,
        "antiguedad_cliente_meses": 24,
        "edad": 34,
        "dispositivos_registrados": 2,
        "porcentaje_uso_app_movil": 65,
        "cantidad_perfiles_creados": 3,
        "interacciones_mensuales_soporte": 1,
        "distancia_promedio_red_km": 12.5,
    }


# Tests del endpoint GET /

def test_endpoint_raiz(client):
    """GET / retorna estado del servicio."""
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()


# Tests del endpoint POST /predict

def test_predict_retorna_cluster(client, usuario_valido):
    """POST /predict retorna un cluster válido."""
    response = client.post("/predict", json=usuario_valido)
    assert response.status_code == 200
    data = response.json()
    assert "cluster" in data
    assert data["cluster"] in [0, 1, 2]


def test_predict_variables_faltantes(client):
    """POST /predict retorna 400 cuando faltan variables."""
    payload_incompleto = {"horas_consumo_mensual": 45, "gasto_mensual": 120}
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 400
    assert "Faltan variables" in response.json()["detail"]


def test_predict_reproducible(client, usuario_valido):
    """Dos predicciones con los mismos datos retornan el mismo cluster."""
    r1 = client.post("/predict", json=usuario_valido)
    r2 = client.post("/predict", json=usuario_valido)
    assert r1.json()["cluster"] == r2.json()["cluster"]


def test_predict_distintos_usuarios(client, usuario_valido):
    """Usuarios con características extremas pueden caer en clusters distintos."""
    usuario_bajo = {k: 1 for k in usuario_valido}
    usuario_alto = {k: 99 for k in usuario_valido}

    r_bajo = client.post("/predict", json=usuario_bajo)
    r_alto = client.post("/predict", json=usuario_alto)

    assert r_bajo.status_code == 200
    assert r_alto.status_code == 200
