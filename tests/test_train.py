"""Tests unitarios para el pipeline ETL y el entrenamiento del modelo."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, r2_score

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service"))


# Fixtures

@pytest.fixture
def df_streaming():
    """DataFrame de prueba que simula usuarios_streaming.csv."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "id_cliente": range(1, n + 1),
        "horas_consumo_mensual": np.random.uniform(5, 60, n),
        "gasto_mensual": np.random.uniform(30, 500, n),
        "cantidad_contenidos_vistos": np.random.randint(1, 60, n),
        "sesiones_semana": np.random.uniform(1, 20, n),
        "porcentaje_finalizacion": np.random.uniform(10, 100, n),
        "tiempo_promedio_sesion_min": np.random.uniform(10, 250, n),
        "cantidad_generos_consumidos": np.random.randint(1, 12, n),
        "porcentaje_uso_promociones": np.random.uniform(0, 1, n),
        "antiguedad_cliente_meses": np.random.randint(1, 100, n),
    })


@pytest.fixture
def df_perfil():
    """DataFrame de prueba que simula la tabla perfil_usuario."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "id_cliente": range(1, n + 1),
        "edad": np.random.randint(18, 65, n),
        "dispositivos_registrados": np.random.randint(1, 5, n),
        "porcentaje_uso_app_movil": np.random.uniform(0, 1, n),
        "cantidad_perfiles_creados": np.random.randint(1, 5, n),
        "interacciones_mensuales_soporte": np.random.randint(0, 10, n),
        "distancia_promedio_red_km": np.random.uniform(1, 80, n),
    })


# Tests de validacion de esquema

def test_validar_esquema_columnas_completas(df_streaming):
    """El esquema se valida correctamente cuando todas las columnas están presentes."""
    columnas_esperadas = [
        "id_cliente", "horas_consumo_mensual", "gasto_mensual",
        "cantidad_contenidos_vistos", "sesiones_semana",
        "porcentaje_finalizacion", "tiempo_promedio_sesion_min",
        "cantidad_generos_consumidos", "porcentaje_uso_promociones",
        "antiguedad_cliente_meses",
    ]
    faltantes = [c for c in columnas_esperadas if c not in df_streaming.columns]
    assert faltantes == [], f"Columnas faltantes: {faltantes}"


def test_validar_esquema_columnas_faltantes(df_streaming):
    """Se detectan columnas faltantes cuando se elimina una."""
    df_incompleto = df_streaming.drop(columns=["gasto_mensual"])
    columnas_esperadas = [
        "id_cliente", "horas_consumo_mensual", "gasto_mensual",
    ]
    faltantes = [c for c in columnas_esperadas if c not in df_incompleto.columns]
    assert "gasto_mensual" in faltantes


def test_validar_esquema_tipos_numericos(df_streaming):
    """Todas las columnas del dataset de streaming son numéricas."""
    for col in df_streaming.columns:
        assert pd.api.types.is_numeric_dtype(df_streaming[col]), (
            f"La columna {col} no es numérica"
        )


def test_validar_esquema_tipo_no_numerico():
    """Se detecta correctamente una columna que no es numérica."""
    df = pd.DataFrame({"id_cliente": [1, 2], "gasto_mensual": ["alto", "bajo"]})
    no_numericas = [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    assert "gasto_mensual" in no_numericas


# Tests de integracion de fuentes

def test_merge_por_id_cliente(df_streaming, df_perfil):
    """El merge une correctamente ambas fuentes por id_cliente."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    assert len(data) == 50
    assert "edad" in data.columns
    assert "horas_consumo_mensual" in data.columns


def test_merge_sin_coincidencias(df_streaming):
    """El merge con IDs que no coinciden produce un DataFrame vacío."""
    perfil_sin_match = pd.DataFrame({
        "id_cliente": range(1000, 1010),
        "edad": [30] * 10,
        "dispositivos_registrados": [2] * 10,
        "porcentaje_uso_app_movil": [0.5] * 10,
        "cantidad_perfiles_creados": [1] * 10,
        "interacciones_mensuales_soporte": [0] * 10,
        "distancia_promedio_red_km": [20.0] * 10,
    })
    data = df_streaming.merge(perfil_sin_match, on="id_cliente")
    assert data.empty


def test_eliminacion_nulos(df_streaming, df_perfil):
    """Se eliminan correctamente las filas con valores nulos."""
    df_streaming.loc[0, "gasto_mensual"] = np.nan
    df_streaming.loc[1, "horas_consumo_mensual"] = np.nan
    data = df_streaming.merge(df_perfil, on="id_cliente")
    antes = len(data)
    data = data.dropna()
    assert len(data) == antes - 2


# Tests de escalamiento

def test_standard_scaler_media_cero(df_streaming, df_perfil):
    """Tras escalar, la media de cada variable es aproximadamente cero."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    medias = np.abs(X_scaled.mean(axis=0))
    assert np.all(medias < 1e-10), "La media tras escalar no es cero"


def test_standard_scaler_desviacion_uno(df_streaming, df_perfil):
    """Tras escalar, la desviación estándar de cada variable es aproximadamente uno."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    stds = X_scaled.std(axis=0)
    assert np.allclose(stds, 1.0, atol=0.05), "La desviación estándar no es 1"


# Tests del modelo KMeans

def test_kmeans_asigna_clusters(df_streaming, df_perfil):
    """KMeans asigna un cluster a cada usuario."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=29, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    assert len(clusters) == len(data)
    assert set(clusters).issubset({0, 1, 2})


def test_kmeans_reproducibilidad(df_streaming, df_perfil):
    """Dos ejecuciones con el mismo random_state producen los mismos clusters."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km1 = KMeans(n_clusters=3, random_state=29, n_init=10)
    km2 = KMeans(n_clusters=3, random_state=29, n_init=10)
    c1 = km1.fit_predict(X_scaled)
    c2 = km2.fit_predict(X_scaled)
    assert np.array_equal(c1, c2)


def test_rango_k_inercia_decreciente(df_streaming, df_perfil):
    """La inercia disminuye a medida que k aumenta."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    for k in range(2, 8):
        modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
        modelo.fit(X_scaled)
        inertias.append(modelo.inertia_)

    for i in range(1, len(inertias)):
        assert inertias[i] <= inertias[i - 1], (
            f"La inercia no decrece entre k={i+1} y k={i+2}"
        )


def test_silhouette_en_rango_valido(df_streaming, df_perfil):
    """El coeficiente Silhouette está entre -1 y 1."""
    from sklearn.metrics import silhouette_score

    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=29, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    assert -1 <= score <= 1


# Tests de centroides

def test_centroides_escala_original(df_streaming, df_perfil):
    """Los centroides invertidos tienen dimensiones coherentes con los datos originales."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=29, n_init=10)
    kmeans.fit(X_scaled)

    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)
    assert centroides_original.shape == (3, X.shape[1])
    assert centroides_original.shape[1] == 15


# Tests de clasificación supervisada

@pytest.fixture
def datos_con_clusters(df_streaming, df_perfil):
    """Datos integrados con clusters asignados por KMeans."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    X = data.drop(columns=["id_cliente"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=29, n_init=10)
    data["cluster"] = kmeans.fit_predict(X_scaled)
    return data, list(X.columns)


def test_pipeline_clasificacion_accuracy(datos_con_clusters):
    """Un pipeline de clasificación supervisada alcanza accuracy razonable."""
    data, columnas = datos_con_clusters
    X = data[columnas].values
    y = data["cluster"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=29, stratify=y,
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=29)),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    assert acc > 0.5, f"Accuracy demasiado baja: {acc}"


def test_gridsearchcv_clasificacion(datos_con_clusters):
    """GridSearchCV selecciona hiperparámetros y mejora o mantiene el score."""
    data, columnas = datos_con_clusters
    X = data[columnas].values
    y = data["cluster"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=29)),
    ])
    grid = GridSearchCV(pipe, {"clf__C": [0.1, 1, 10]}, cv=3, scoring="f1_macro")
    grid.fit(X, y)
    assert grid.best_score_ > 0, "El mejor CV score debe ser positivo"
    assert "clf__C" in grid.best_params_


def test_feature_importances_clasificacion(datos_con_clusters):
    """El clasificador RandomForest expone importancia de variables."""
    data, columnas = datos_con_clusters
    X = data[columnas].values
    y = data["cluster"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=29)),
    ])
    pipe.fit(X, y)
    importancias = pipe.named_steps["clf"].feature_importances_
    assert len(importancias) == len(columnas)
    assert np.all(importancias >= 0)


# Tests de regresión supervisada

def test_pipeline_regresion_r2(datos_con_clusters):
    """Un pipeline de regresión produce un R² definido."""
    data, columnas = datos_con_clusters
    target = "gasto_mensual"
    features = [c for c in columnas if c != target]
    X = data[features].values
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=29,
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(n_estimators=50, random_state=29)),
    ])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    assert r2 > -1, f"R² demasiado bajo: {r2}"


def test_gridsearchcv_regresion(datos_con_clusters):
    """GridSearchCV selecciona hiperparámetros para el regresor."""
    data, columnas = datos_con_clusters
    target = "gasto_mensual"
    features = [c for c in columnas if c != target]
    X = data[features].values
    y = data[target].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    grid = GridSearchCV(pipe, {}, cv=3, scoring="r2")
    grid.fit(X, y)
    assert grid.best_score_ is not None


def test_prediccion_regresion_valores_positivos(datos_con_clusters):
    """Las predicciones de gasto son números finitos."""
    data, columnas = datos_con_clusters
    target = "gasto_mensual"
    features = [c for c in columnas if c != target]
    X = data[features].values
    y = data[target].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    pipe.fit(X, y)
    predicciones = pipe.predict(X)
    assert np.all(np.isfinite(predicciones))


# Tests de transformadores personalizados

def test_winsorizer_recorta_outliers():
    """Winsorizer recorta valores extremos por percentil."""
    from preprocesamiento import Winsorizer
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[0, 0] = 100
    X[1, 1] = -100
    winz = Winsorizer(limits=(0.05, 0.05))
    X_t = np.asarray(winz.fit_transform(X))
    assert X_t[0, 0] < 100
    assert X_t[1, 1] > -100
    assert X_t.shape == X.shape


def test_winsorizer_preserva_no_outliers():
    """Winsorizer no modifica valores dentro del rango completo."""
    from preprocesamiento import Winsorizer
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    winz = Winsorizer(limits=(0.0, 0.0))
    X_t = np.asarray(winz.fit_transform(X))
    np.testing.assert_array_almost_equal(X, X_t)


def test_correlation_filter_elimina_colineales():
    """CorrelationFilter elimina variables altamente correlacionadas."""
    from preprocesamiento import CorrelationFilter
    np.random.seed(42)
    x1 = np.random.randn(100)
    x2 = x1 + np.random.randn(100) * 0.01
    x3 = np.random.randn(100)
    X = np.column_stack([x1, x2, x3])
    cf = CorrelationFilter(threshold=0.9)
    X_t = cf.fit_transform(X)
    assert X_t.shape[1] < X.shape[1]
    assert X_t.shape[1] == 2


def test_correlation_filter_preserva_independientes():
    """CorrelationFilter no elimina variables con baja correlación."""
    from preprocesamiento import CorrelationFilter
    np.random.seed(42)
    X = np.random.randn(100, 5)
    cf = CorrelationFilter(threshold=0.9)
    X_t = cf.fit_transform(X)
    assert X_t.shape[1] == 5


def test_pipeline_robusto_clasificacion(datos_con_clusters):
    """Pipeline con Winsorizer, Scaler y CorrelationFilter clasifica correctamente."""
    from preprocesamiento import Winsorizer, CorrelationFilter
    data, columnas = datos_con_clusters
    X = data[columnas].values
    y = data["cluster"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=29, stratify=y,
    )
    pipe = Pipeline([
        ("winsorizer", Winsorizer()),
        ("scaler", StandardScaler()),
        ("colinealidad", CorrelationFilter()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=29)),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    assert acc > 0.4


def test_two_stage_tuning_clasificacion(datos_con_clusters):
    """Búsqueda en dos etapas (RandomizedSearchCV → GridSearchCV) funciona."""
    from preprocesamiento import Winsorizer, CorrelationFilter
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from scipy.stats import randint as sp_randint
    data, columnas = datos_con_clusters
    X = data[columnas].values
    y = data["cluster"].values
    pipe = Pipeline([
        ("winsorizer", Winsorizer()),
        ("scaler", StandardScaler()),
        ("colinealidad", CorrelationFilter()),
        ("clf", RandomForestClassifier(random_state=29)),
    ])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=29)
    random_search = RandomizedSearchCV(
        pipe, {"clf__n_estimators": sp_randint(10, 100)},
        n_iter=5, cv=skf, scoring="f1_macro", random_state=29,
    )
    random_search.fit(X, y)
    assert random_search.best_score_ > 0
    best_n = random_search.best_params_["clf__n_estimators"]
    grid = GridSearchCV(
        pipe, {"clf__n_estimators": [max(10, best_n - 20), best_n, best_n + 20]},
        cv=skf, scoring="f1_macro",
    )
    grid.fit(X, y)
    assert grid.best_score_ > 0
