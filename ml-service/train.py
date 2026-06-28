import pandas as pd
import json
import pickle
import os
import sys

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kneed import KneeLocator

DB_USER = os.environ["POSTGRES_USER"]
DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]
DB_NAME = os.environ["POSTGRES_DB"]

# Esquemas esperados por fuente: columnas obligatorias del conjunto
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


def validar_esquema(df, columnas_esperadas, nombre_fuente):
    """Valida que la fuente tenga las columnas esperadas y que sean numéricas."""
    faltantes = [c for c in columnas_esperadas if c not in df.columns]
    if faltantes:
        print(f"Error: {nombre_fuente} no contiene las columnas: {faltantes}")
        sys.exit(1)

    no_numericas = [
        c for c in columnas_esperadas if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if no_numericas:
        print(f"Error: en {nombre_fuente} estas columnas no son numéricas: {no_numericas}")
        sys.exit(1)

    print(f"Esquema de {nombre_fuente} validado correctamente.")


os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Consumo dentro de la plataforma
try:
    streaming = pd.read_csv("data/raw/usuarios_streaming.csv")
    print(f"usuarios_streaming.csv cargado: {len(streaming)} registros")
except FileNotFoundError:
    print("Error: no se encontró data/raw/usuarios_streaming.csv")
    sys.exit(1)

validar_esquema(streaming, COLUMNAS_STREAMING, "usuarios_streaming.csv")

# Perfil del usuario
try:
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@postgres:5432/{DB_NAME}")
    perfil = pd.read_sql("SELECT * FROM perfil_usuario", engine)
    print(f"perfil_usuario cargado desde PostgreSQL: {len(perfil)} registros")
except Exception as e:
    print(f"Error al conectar con PostgreSQL: {e}")
    sys.exit(1)

validar_esquema(perfil, COLUMNAS_PERFIL, "perfil_usuario")

# Integración por id_cliente
data = streaming.merge(perfil, on="id_cliente")

if data.empty:
    print("Error: el merge entre fuentes no produjo registros. Verificar id_cliente.")
    sys.exit(1)

if data.isnull().any().any():
    nulos = data.isnull().sum().sum()
    print(f"Advertencia: se encontraron {nulos} valores nulos. Se eliminan filas afectadas.")
    data = data.dropna()

print(f"Conjunto integrado: {len(data)} registros, {len(data.columns)} variables")
data.to_csv("data/processed/usuarios_integrados.csv", index=False)

# Matriz de variables sin id
X = data.drop(columns=["id_cliente"])

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluación de k usando metodo del codo y silhouettes
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
    print("Advertencia: KneeLocator no detectó un codo claro. Se usa k=3 por defecto.")
    k_optimo = 3
else:
    k_optimo = int(kl.elbow)

# Entrenamiento final
kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data["cluster"] = clusters

print(f"Modelo entrenado con k_optimo = {k_optimo}")

# PCA
pca = PCA(n_components=2)
componentes = pca.fit_transform(X_scaled)
data["pc1"] = componentes[:, 0]
data["pc2"] = componentes[:, 1]

data.to_csv("outputs/usuarios_segmentados.csv", index=False)

# Métricas
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

# Centroides en escala original
centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroides_df = pd.DataFrame(centroides_original, columns=X.columns)
centroides_df.to_csv("outputs/centroides.csv", index=False)

# Persistencia
pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(pca, open("models/pca.pkl", "wb"))

print("Modelo, escalador y métricas guardados.")
