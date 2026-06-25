import pandas as pd
import json
import pickle
import os

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kneed import KneeLocator

os.makedirs("models", exist_ok=True)

# Consumo dentro de la plataforma
streaming = pd.read_csv("data/usuarios_streaming.csv")

# Perfil del usuario
engine = create_engine("postgresql://admin:admin@postgres:5432/streaming_usuarios")
perfil = pd.read_sql("SELECT * FROM perfil_usuario", engine)

# Integración por id_cliente
data = streaming.merge(perfil, on="id_cliente")
data.to_csv("data/usuarios_integrados.csv", index=False)

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

data.to_csv("data/usuarios_segmentados.csv", index=False)

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
centroides_df.to_csv("data/centroides.csv", index=False)

# Persistencia
pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(pca, open("models/pca.pkl", "wb"))

print("Modelo, escalador y métricas guardados.")
