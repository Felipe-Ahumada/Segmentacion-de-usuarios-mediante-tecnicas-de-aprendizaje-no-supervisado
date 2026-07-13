"""Pipeline ETL y entrenamiento de modelos de segmentación y predicción.

Integra el CSV de consumo con la tabla perfil_usuario de PostgreSQL, valida el
esquema de cada fuente, selecciona k con el método del codo y el coeficiente
Silhouette, entrena KMeans, y aplica modelos supervisados (clasificación y
regresión) con un pipeline robusto de preprocesamiento y optimización en dos
etapas (RandomizedSearchCV → GridSearchCV).
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.stats import loguniform, randint as sp_randint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sqlalchemy import create_engine

from preprocesamiento import CorrelationFilter, Winsorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_USER = os.environ["POSTGRES_USER"]
DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]
DB_NAME = os.environ["POSTGRES_DB"]

TAMANO_BLOQUE = 10000

COLUMNAS_STREAMING = [
    "id_cliente",
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
]
COLUMNAS_PERFIL = [
    "id_cliente",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km",
]


# ─── Funciones auxiliares ────────────────────────────────────────────────


def validar_esquema(df, columnas_esperadas, nombre_fuente):
    """Valida que la fuente tenga las columnas esperadas y que sean numéricas."""
    faltantes = [c for c in columnas_esperadas if c not in df.columns]
    if faltantes:
        logger.error("%s no contiene las columnas: %s", nombre_fuente, faltantes)
        sys.exit(1)

    no_numericas = [
        c for c in columnas_esperadas if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if no_numericas:
        logger.error("En %s estas columnas no son numéricas: %s", nombre_fuente, no_numericas)
        sys.exit(1)

    logger.info("Esquema de %s validado correctamente.", nombre_fuente)


def _param_nativo(v):
    """Convierte tipos NumPy a nativos de Python para serialización JSON."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _refinar_params(best_params):
    """Genera grilla refinada alrededor de los mejores hiperparámetros.

    Para valores numéricos enteros genera 5 candidatos con paso proporcional.
    Para flotantes, multiplica por factores entre 0.5 y 1.5.
    Cadenas y None se mantienen fijos.
    """
    refinados = {}
    for key, val in best_params.items():
        if val is None:
            refinados[key] = [None]
        elif isinstance(val, str):
            refinados[key] = [val]
        elif isinstance(val, (int, np.integer)):
            v = int(val)
            step = max(1, v // 4)
            mn = 2 if "min_samples_split" in key else 1
            refinados[key] = sorted(set(
                max(mn, v + d * step) for d in (-2, -1, 0, 1, 2)
            ))
        elif isinstance(val, (float, np.floating)):
            v = float(val)
            refinados[key] = sorted(set(
                round(max(1e-8, v * f), 8) for f in (0.5, 0.75, 1.0, 1.25, 1.5)
            ))
        else:
            refinados[key] = [val]
    return refinados


def _extraer_importancia(pipeline, nombres_features, step_modelo, step_filtro):
    """Extrae la importancia de variables del modelo final del pipeline.

    Considera que CorrelationFilter puede haber eliminado columnas,
    por lo que mapea los índices sobrevivientes a los nombres originales.
    """
    modelo = pipeline.named_steps[step_modelo]
    corr = pipeline.named_steps[step_filtro]
    n = len(nombres_features)
    kept = [i for i in range(n) if i not in corr.columns_to_drop_]
    nombres_post = [nombres_features[i] for i in kept]

    if hasattr(modelo, "feature_importances_"):
        return dict(zip(
            nombres_post, [round(float(v), 4) for v in modelo.feature_importances_],
        )), nombres_post
    if hasattr(modelo, "coef_"):
        coefs = np.abs(modelo.coef_)
        if coefs.ndim > 1:
            coefs = coefs.mean(axis=0)
        return dict(zip(
            nombres_post, [round(float(v), 4) for v in coefs],
        )), nombres_post
    return None, nombres_post


def _validar_rangos(df):
    """Valida y corrige columnas porcentuales al rango [0, 100]."""
    columnas_pct = [
        c for c in [
            "porcentaje_finalizacion",
            "porcentaje_uso_promociones",
            "porcentaje_uso_app_movil",
        ]
        if c in df.columns
    ]
    for col in columnas_pct:
        fuera = ~df[col].between(0, 100)
        if fuera.any():
            logger.warning(
                "%s: %d valores fuera de [0, 100]. Recortados con clip.", col, fuera.sum(),
            )
            df[col] = df[col].clip(0, 100)
    return df


# ─── ETL ─────────────────────────────────────────────────────────────────


def cargar_datos():
    """Carga, valida e integra las fuentes CSV y PostgreSQL.

    Lee usuarios_streaming.csv por bloques y la tabla perfil_usuario desde
    PostgreSQL. Valida esquemas, realiza el merge por id_cliente, elimina
    nulos y persiste el resultado integrado.

    Returns:
        pd.DataFrame: Datos integrados listos para modelar.
    """
    try:
        bloques = pd.read_csv(
            "data/raw/usuarios_streaming.csv", chunksize=TAMANO_BLOQUE,
        )
        streaming = pd.concat(bloques, ignore_index=True)
        logger.info("usuarios_streaming.csv cargado: %d registros", len(streaming))
    except FileNotFoundError:
        logger.error("No se encontró data/raw/usuarios_streaming.csv")
        sys.exit(1)

    validar_esquema(streaming, COLUMNAS_STREAMING, "usuarios_streaming.csv")

    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASSWORD}@postgres:5432/{DB_NAME}",
        )
        bloques_perfil = pd.read_sql(
            "SELECT * FROM perfil_usuario", engine, chunksize=TAMANO_BLOQUE,
        )
        perfil = pd.concat(bloques_perfil, ignore_index=True)
        logger.info("perfil_usuario cargado desde PostgreSQL: %d registros", len(perfil))
    except Exception as e:
        logger.error("Error al conectar con PostgreSQL: %s", e)
        sys.exit(1)

    validar_esquema(perfil, COLUMNAS_PERFIL, "perfil_usuario")

    data = streaming.merge(perfil, on="id_cliente", how="inner", validate="one_to_one")

    if data.empty:
        logger.error("El merge entre fuentes no produjo registros. Verificar id_cliente.")
        sys.exit(1)

    total_antes = len(data)
    nulos_por_col = data.isnull().sum()
    nulos_existentes = nulos_por_col[nulos_por_col > 0]
    if not nulos_existentes.empty:
        filas_con_nulo = data.isnull().any(axis=1).sum()
        pct_filas = filas_con_nulo / total_antes * 100
        logger.warning(
            "Nulos detectados: %d valores en %d filas (%.1f%% del dataset).",
            nulos_existentes.sum(), filas_con_nulo, pct_filas,
        )
        logger.info("Detalle por columna: %s", nulos_existentes.to_dict())
        # Se usa dropna porque las filas afectadas son una fracción menor del
        # dataset (<5 %).  Imputar requeriría supuestos sobre el mecanismo de
        # ausencia (MCAR/MAR) que no se pueden validar con la información
        # disponible; eliminar preserva la integridad de las relaciones entre
        # variables sin introducir sesgo artificial.
        data = data.dropna()
        logger.info(
            "Filas retenidas tras limpieza: %d de %d (%.1f%%).",
            len(data), total_antes, len(data) / total_antes * 100,
        )
    else:
        logger.info("Sin valores nulos. No se requiere limpieza adicional.")

    # Validación de rangos porcentuales (.pipe para encadenar transformaciones)
    data = data.pipe(_validar_rangos)

    # Detección de outliers por IQR (informativo; tratamiento en pipeline con Winsorizer)
    numericas = data.select_dtypes(include=[np.number]).columns.drop("id_cliente")
    outliers_iqr = {}
    for col in numericas:
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            outliers_iqr[col] = n_out
    if outliers_iqr:
        logger.info(
            "Outliers por IQR (tratados con Winsorizer en pipeline): %s", outliers_iqr,
        )

    # Optimización de tipos para reducir uso de memoria
    mem_antes = data.memory_usage(deep=True).sum()
    for col in data.select_dtypes(include=["int64"]).columns:
        data[col] = pd.to_numeric(data[col], downcast="integer")
    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = pd.to_numeric(data[col], downcast="float")
    mem_despues = data.memory_usage(deep=True).sum()
    logger.info(
        "Memoria optimizada: %.1f KB → %.1f KB (%.0f%% reducción).",
        mem_antes / 1024, mem_despues / 1024,
        (1 - mem_despues / mem_antes) * 100,
    )

    # Filtro compuesto: detectar usuarios con alto gasto pero baja finalización
    q75_gasto = data["gasto_mensual"].quantile(0.75)
    q25_final = data["porcentaje_finalizacion"].quantile(0.25)
    anomalos = data.query(
        "gasto_mensual > @q75_gasto and porcentaje_finalizacion < @q25_final"
    )
    if len(anomalos) > 0:
        logger.info(
            "Usuarios con alto gasto (>$%.0f) y baja finalización (<%.0f%%): %d (%.1f%%).",
            q75_gasto, q25_final, len(anomalos), len(anomalos) / len(data) * 100,
        )

    # Resumen estadístico agrupado por rangos de antigüedad (.assign + groupby)
    resumen = (
        data
        .assign(rango_antiguedad=lambda df: pd.cut(
            df["antiguedad_cliente_meses"],
            bins=[0, 12, 36, 72, float("inf")],
            labels=["Nuevo", "Regular", "Antiguo", "Veterano"],
        ))
        .groupby("rango_antiguedad", observed=True)
        .agg(
            gasto_promedio=("gasto_mensual", "mean"),
            horas_promedio=("horas_consumo_mensual", "mean"),
            sesiones_promedio=("sesiones_semana", "mean"),
            usuarios=("id_cliente", "count"),
        )
        .round(2)
    )
    logger.info("Resumen por antigüedad:\n%s", resumen.to_string())

    # Resumen de calidad post-limpieza
    logger.info(
        "Calidad post-limpieza: %d registros, %d variables, "
        "%d nulos, %d duplicados en variables clave.",
        len(data), len(data.columns),
        int(data.isnull().sum().sum()),
        int(data.duplicated(subset=[c for c in data.columns if c != "id_cliente"]).sum()),
    )

    logger.info("Conjunto integrado: %d registros, %d variables.", len(data), len(data.columns))
    data.to_csv("data/processed/usuarios_integrados.csv", index=False)
    return data


# ─── Aprendizaje no supervisado ──────────────────────────────────────────


def entrenar_kmeans(data):
    """Selecciona k óptimo con el método del codo, entrena KMeans y aplica PCA.

    Evalúa k en el rango [2, 10] usando inercia y coeficiente Silhouette.
    Persiste el modelo, el scaler, PCA, métricas y centroides.

    Args:
        data: DataFrame integrado con todas las variables y id_cliente.

    Returns:
        tuple: (data con columnas cluster/pc1/pc2, lista de nombres de columnas).
    """
    X = data.drop(columns=["id_cliente"])
    columnas_modelo = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouettes = []
    rango_k = range(2, 11)
    for k in rango_k:
        modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
        modelo.fit(X_scaled)
        inertias.append(modelo.inertia_)
        silhouettes.append(silhouette_score(X_scaled, modelo.labels_))

    kl = KneeLocator(rango_k, inertias, curve="convex", direction="decreasing")

    if kl.elbow is None:
        logger.warning("KneeLocator no detectó un codo claro. Se usa k=3 por defecto.")
        k_optimo = 3
    else:
        k_optimo = int(kl.elbow)

    kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)
    data["cluster"] = kmeans.fit_predict(X_scaled)
    logger.info("Modelo entrenado con k_optimo = %d", k_optimo)

    # Tabla pivote: perfil estadístico por cluster
    pivot = data.pivot_table(
        values=["gasto_mensual", "horas_consumo_mensual", "sesiones_semana"],
        index="cluster",
        aggfunc=["mean", "std"],
    ).round(2)
    logger.info("Perfil estadístico por cluster:\n%s", pivot.to_string())

    # Broadcasting + reshape: distancia de cada usuario a todos los centroides
    X_3d = X_scaled.reshape(len(data), 1, -1)
    centros_3d = kmeans.cluster_centers_.reshape(1, k_optimo, -1)
    diff = X_3d - centros_3d  # broadcasting: (n,1,p) - (1,k,p) → (n,k,p)
    distancias_todos = np.sqrt((diff ** 2).sum(axis=2))  # (n, k)
    data["distancia_centroide"] = distancias_todos[
        np.arange(len(data)), data["cluster"].values
    ]
    logger.info(
        "Distancia promedio al centroide por cluster: %s",
        data.groupby("cluster")["distancia_centroide"].mean().round(3).to_dict(),
    )

    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_scaled)
    data["pc1"] = componentes[:, 0]
    data["pc2"] = componentes[:, 1]

    data.to_csv("outputs/usuarios_segmentados.csv", index=False)

    metricas = {
        "k_optimo": k_optimo,
        "silhouette_score": float(silhouette_score(X_scaled, data["cluster"])),
        "n_usuarios": int(len(data)),
        "n_clusters": k_optimo,
        "varianza_pca": float(pca.explained_variance_ratio_.sum()),
        "inertias": [float(v) for v in inertias],
        "silhouettes": [float(v) for v in silhouettes],
        "rango_k": list(rango_k),
    }
    with open("models/metricas.json", "w") as f:
        json.dump(metricas, f, indent=4)

    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centroides_df = pd.DataFrame(centroides_original, columns=columnas_modelo)
    centroides_df.to_csv("outputs/centroides.csv", index=False)

    pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(pca, open("models/pca.pkl", "wb"))

    logger.info("Modelo, escalador y métricas guardados.")
    return data, columnas_modelo


# ─── Aprendizaje supervisado ─────────────────────────────────────────────


def entrenar_supervisado(data, columnas_modelo):
    """Entrena clasificadores y regresores con pipeline robusto en dos etapas.

    Pipeline de preprocesamiento: Winsorizer → StandardScaler → CorrelationFilter.
    Optimización: RandomizedSearchCV (exploración) → GridSearchCV (refinamiento).
    Validación cruzada estratificada para clasificación, KFold para regresión.

    Persiste el mejor clasificador, el mejor regresor y las métricas completas.

    Args:
        data: DataFrame con columna 'cluster' asignada por KMeans.
        columnas_modelo: Lista de nombres de features (sin id_cliente).
    """
    logger.info("Iniciando entrenamiento de modelos supervisados.")

    data_sup = data.drop_duplicates(subset=columnas_modelo)
    n_dup = len(data) - len(data_sup)
    if n_dup > 0:
        logger.info("Duplicados eliminados: %d registros", n_dup)
    else:
        data_sup = data

    resultados_cls, mejor_clasificador, mejor_nombre_cls, nombres_cls_post = (
        _entrenar_clasificadores(data_sup, columnas_modelo)
    )
    importancia_cls, nombres_cls_post = _extraer_importancia(
        mejor_clasificador, columnas_modelo, "clf", "colinealidad",
    )

    resultados_reg, mejor_regresor, mejor_nombre_reg, y_test_r, y_pred_r, features_reg = (
        _entrenar_regresores(data_sup, columnas_modelo)
    )
    importancia_reg, nombres_reg_post = _extraer_importancia(
        mejor_regresor, features_reg, "reg", "colinealidad",
    )

    metricas_supervisado = {
        "clasificacion": {
            "resultados": resultados_cls,
            "mejor_modelo": mejor_nombre_cls,
            "importancia_variables": importancia_cls,
            "columnas": columnas_modelo,
            "features_post_filtro": nombres_cls_post,
        },
        "regresion": {
            "target": "gasto_mensual",
            "features": features_reg,
            "features_post_filtro": nombres_reg_post,
            "resultados": resultados_reg,
            "mejor_modelo": mejor_nombre_reg,
            "importancia_variables": importancia_reg,
            "y_test": [round(float(v), 2) for v in y_test_r],
            "y_pred": [round(float(v), 2) for v in y_pred_r],
        },
    }

    with open("models/metricas_supervisado.json", "w") as f:
        json.dump(metricas_supervisado, f, indent=4)

    pickle.dump(mejor_clasificador, open("models/mejor_clasificador.pkl", "wb"))
    pickle.dump(mejor_regresor, open("models/mejor_regresor.pkl", "wb"))

    logger.info("Modelos supervisados y métricas guardados.")


def _entrenar_clasificadores(data_sup, columnas_modelo):
    """Entrena 4 clasificadores con pipeline robusto y tuning en dos etapas.

    Returns:
        tuple: (resultados_dict, mejor_pipeline, nombre_mejor, features_post_filtro).
    """
    y_cls = data_sup["cluster"].values
    X_cls = data_sup[columnas_modelo].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=29, stratify=y_cls,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=29)

    clasificadores = {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
                ("colinealidad", CorrelationFilter()),
                ("clf", LogisticRegression(max_iter=1000, random_state=29)),
            ]),
            "params_random": {
                "clf__C": loguniform(0.01, 100),
                "clf__solver": ["lbfgs", "saga"],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
                ("colinealidad", CorrelationFilter()),
                ("clf", RandomForestClassifier(random_state=29)),
            ]),
            "params_random": {
                "clf__n_estimators": sp_randint(50, 300),
                "clf__max_depth": [None, 5, 10, 15, 20],
                "clf__min_samples_split": sp_randint(2, 20),
                "clf__min_samples_leaf": sp_randint(1, 10),
            },
        },
        "SVM": {
            "pipeline": Pipeline([
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
                ("colinealidad", CorrelationFilter()),
                ("clf", SVC(random_state=29)),
            ]),
            "params_random": {
                "clf__C": loguniform(0.01, 100),
                "clf__kernel": ["rbf", "linear", "poly"],
                "clf__gamma": ["scale", "auto"],
            },
        },
        "KNN": {
            "pipeline": Pipeline([
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
                ("colinealidad", CorrelationFilter()),
                ("clf", KNeighborsClassifier()),
            ]),
            "params_random": {
                "clf__n_neighbors": sp_randint(3, 20),
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        },
    }

    resultados = {}
    mejor_f1 = 0
    mejor_pipeline = None
    mejor_nombre = None

    for nombre, cfg in clasificadores.items():
        logger.info("Clasificador %s — Etapa 1: RandomizedSearchCV", nombre)
        random_search = RandomizedSearchCV(
            cfg["pipeline"], cfg["params_random"],
            n_iter=20, cv=skf, scoring="f1_macro",
            n_jobs=-1, random_state=29,
        )
        random_search.fit(X_train, y_train)
        stage1_score = float(random_search.best_score_)
        logger.info("  Etapa 1 — CV score: %.4f", stage1_score)

        params_refinados = _refinar_params(random_search.best_params_)
        logger.info("Clasificador %s — Etapa 2: GridSearchCV refinado", nombre)
        grid = GridSearchCV(
            cfg["pipeline"], params_refinados,
            cv=skf, scoring="f1_macro", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        resultados[nombre] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "best_params": {k: _param_nativo(v) for k, v in grid.best_params_.items()},
            "cv_score": round(float(grid.best_score_), 4),
            "stage1_score": round(stage1_score, 4),
        }

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_pipeline = grid.best_estimator_
            mejor_nombre = nombre

        logger.info("Clasificador %s — Accuracy: %.3f, F1 macro: %.3f", nombre, acc, f1)

    logger.info("Mejor clasificador: %s (F1 = %.3f)", mejor_nombre, mejor_f1)

    corr_cls = mejor_pipeline.named_steps["colinealidad"]
    n = len(columnas_modelo)
    kept = [i for i in range(n) if i not in corr_cls.columns_to_drop_]
    nombres_post = [columnas_modelo[i] for i in kept]

    return resultados, mejor_pipeline, mejor_nombre, nombres_post


def _entrenar_regresores(data_sup, columnas_modelo):
    """Entrena 3 regresores con pipeline robusto y tuning en dos etapas.

    Returns:
        tuple: (resultados_dict, mejor_pipeline, nombre_mejor,
                y_test, y_pred_mejor, features_reg).
    """
    # Justificación estadística de la variable objetivo
    corr_abs = data_sup[columnas_modelo].corr().abs()
    np.fill_diagonal(corr_abs.values, 0)
    promedio_corr = corr_abs.mean().sort_values(ascending=False)
    logger.info(
        "Correlación media absoluta por variable:\n%s",
        promedio_corr.round(4).to_string(),
    )
    logger.info(
        "Variable con mayor correlación promedio: %s (%.4f). "
        "Se selecciona 'gasto_mensual' como target de regresión por combinar "
        "alta predictibilidad estadística con relevancia de negocio "
        "(proxy directo del valor económico del cliente).",
        promedio_corr.index[0], promedio_corr.iloc[0],
    )

    target_reg = "gasto_mensual"
    features_reg = [c for c in columnas_modelo if c != target_reg]
    X_reg = data_sup[features_reg].values
    y_reg = data_sup[target_reg].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=29,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=29)

    regresores = {
        "Linear Regression": {
            "pipeline": Pipeline([
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
                ("colinealidad", CorrelationFilter()),
                ("reg", LinearRegression()),
            ]),
            "params_random": None,
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
                ("colinealidad", CorrelationFilter()),
                ("reg", RandomForestRegressor(random_state=29)),
            ]),
            "params_random": {
                "reg__n_estimators": sp_randint(50, 300),
                "reg__max_depth": [None, 5, 10, 15, 20],
                "reg__min_samples_split": sp_randint(2, 20),
                "reg__min_samples_leaf": sp_randint(1, 10),
            },
        },
        "Gradient Boosting": {
            "pipeline": Pipeline([
                ("winsorizer", Winsorizer()),
                ("scaler", StandardScaler()),
                ("colinealidad", CorrelationFilter()),
                ("reg", GradientBoostingRegressor(random_state=29)),
            ]),
            "params_random": {
                "reg__n_estimators": sp_randint(50, 300),
                "reg__learning_rate": loguniform(0.01, 0.3),
                "reg__max_depth": [3, 5, 7, 10],
                "reg__min_samples_split": sp_randint(2, 20),
            },
        },
    }

    resultados = {}
    mejor_r2 = -float("inf")
    mejor_pipeline = None
    mejor_nombre = None

    for nombre, cfg in regresores.items():
        if cfg["params_random"] is not None:
            logger.info("Regresor %s — Etapa 1: RandomizedSearchCV", nombre)
            random_search = RandomizedSearchCV(
                cfg["pipeline"], cfg["params_random"],
                n_iter=20, cv=kf, scoring="r2",
                n_jobs=-1, random_state=29,
            )
            random_search.fit(X_train, y_train)
            stage1_score = float(random_search.best_score_)
            logger.info("  Etapa 1 — CV score: %.4f", stage1_score)

            params_refinados = _refinar_params(random_search.best_params_)
            logger.info("Regresor %s — Etapa 2: GridSearchCV refinado", nombre)
            grid = GridSearchCV(
                cfg["pipeline"], params_refinados,
                cv=kf, scoring="r2", n_jobs=-1,
            )
            grid.fit(X_train, y_train)
        else:
            logger.info("Regresor %s — Entrenamiento directo (sin hiperparámetros)", nombre)
            grid = GridSearchCV(
                cfg["pipeline"], {},
                cv=kf, scoring="r2", n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            stage1_score = float(grid.best_score_)

        y_pred = grid.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        resultado = {
            "r2": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "best_params": {k: _param_nativo(v) for k, v in grid.best_params_.items()},
            "cv_score": round(float(grid.best_score_), 4),
        }
        if cfg["params_random"] is not None:
            resultado["stage1_score"] = round(stage1_score, 4)

        resultados[nombre] = resultado

        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_pipeline = grid.best_estimator_
            mejor_nombre = nombre

        logger.info("Regresor %s — R²: %.3f, MAE: %.2f, RMSE: %.2f", nombre, r2, mae, rmse)

    logger.info("Mejor regresor: %s (R² = %.3f)", mejor_nombre, mejor_r2)

    y_pred_mejor = mejor_pipeline.predict(X_test)
    return resultados, mejor_pipeline, mejor_nombre, y_test, y_pred_mejor, features_reg


# ─── Punto de entrada ────────────────────────────────────────────────────


def main():
    """Ejecuta el pipeline completo: ETL → KMeans → supervisado."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    data = cargar_datos()
    data, columnas_modelo = entrenar_kmeans(data)
    entrenar_supervisado(data, columnas_modelo)


if __name__ == "__main__":
    main()
