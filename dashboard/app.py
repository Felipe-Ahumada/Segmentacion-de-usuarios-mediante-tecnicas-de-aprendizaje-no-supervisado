import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Segmentación de Usuarios Streaming", layout="wide")
st.title("Segmentación de Usuarios — Plataforma de Streaming")

# Datos desde el servicio ML
respuesta = requests.get("http://ml-service:8000/dashboard-data")
payload = respuesta.json()

usuarios = pd.DataFrame(payload["usuarios"])
centroides = pd.DataFrame(payload["centroides"])
metricas = payload["metricas"]

# Métricas globales
st.subheader("Métricas del modelo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clusters (k óptimo)", metricas["k_optimo"])
col2.metric("Usuarios", metricas["n_usuarios"])
col3.metric("Silhouette Score", f"{metricas['silhouette_score']:.3f}")
col4.metric("Varianza PCA (2D)", f"{metricas['varianza_pca']:.1%}")

# Filtros interactivos
st.sidebar.header("Filtros")
clusters_disponibles = sorted(usuarios["cluster"].unique().tolist())
clusters_sel = st.sidebar.multiselect(
    "Segmentos a incluir",
    clusters_disponibles,
    default=clusters_disponibles,
)
antiguedad_min, antiguedad_max = st.sidebar.slider(
    "Antigüedad (meses)",
    int(usuarios["antiguedad_cliente_meses"].min()),
    int(usuarios["antiguedad_cliente_meses"].max()),
    (
        int(usuarios["antiguedad_cliente_meses"].min()),
        int(usuarios["antiguedad_cliente_meses"].max()),
    ),
)

datos = usuarios[
    usuarios["cluster"].isin(clusters_sel)
    & usuarios["antiguedad_cliente_meses"].between(antiguedad_min, antiguedad_max)
]

# ============================================================
# 1. Visualización general de segmentos
# ============================================================
st.header("1. Visualización general de segmentos")

conteo = datos["cluster"].value_counts().sort_index()
porcentaje = (conteo / conteo.sum() * 100).round(2)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Cantidad de usuarios por cluster")
    st.bar_chart(conteo)
with col_b:
    st.subheader("Distribución porcentual")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        porcentaje,
        labels=[f"Cluster {c}" for c in porcentaje.index],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Porcentaje de usuarios por segmento")
    st.pyplot(fig)

resumen_tamano = pd.DataFrame({"usuarios": conteo, "porcentaje (%)": porcentaje})
st.dataframe(resumen_tamano)

# ============================================================
# 2. Perfilamiento de segmentos
# ============================================================
st.header("2. Perfilamiento de segmentos")

variables_perfil = [
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "antiguedad_cliente_meses",
    "porcentaje_uso_promociones",
    "dispositivos_registrados",
    "edad",
    "porcentaje_uso_app_movil",
    "interacciones_mensuales_soporte",
]
perfil_segmentos = datos.groupby("cluster")[variables_perfil].mean().round(2)
st.dataframe(perfil_segmentos)

# ============================================================
# 3. Comparación entre segmentos
# ============================================================
st.header("3. Comparación entre segmentos")

# 3.a Mapa de calor de promedios normalizados
st.subheader("Mapa de calor — promedios normalizados por cluster")
normalizado = (perfil_segmentos - perfil_segmentos.min()) / (
    perfil_segmentos.max() - perfil_segmentos.min()
)
fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(perfil_segmentos))))
sns.heatmap(normalizado, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
ax.set_xlabel("Variable")
ax.set_ylabel("Cluster")
st.pyplot(fig)

# 3.b Gráfico radial (radar) por segmento
st.subheader("Gráfico radial por segmento")
categorias = variables_perfil
N = len(categorias)
angulos = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angulos += angulos[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for cluster_id in normalizado.index:
    valores = normalizado.loc[cluster_id].tolist()
    valores += valores[:1]
    ax.plot(angulos, valores, label=f"Cluster {cluster_id}")
    ax.fill(angulos, valores, alpha=0.1)

ax.set_xticks(angulos[:-1])
ax.set_xticklabels(categorias, fontsize=8)
ax.set_yticklabels([])
ax.set_title("Perfil comparativo de segmentos (escala normalizada)")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig)

# 3.c Distribución por variable seleccionable
st.subheader("Distribución por variable")
variable_dist = st.selectbox("Variable a comparar", variables_perfil)
fig, ax = plt.subplots(figsize=(10, 5))
for cluster_id in sorted(datos["cluster"].unique()):
    subset = datos[datos["cluster"] == cluster_id][variable_dist]
    ax.hist(subset, bins=15, alpha=0.5, label=f"Cluster {cluster_id}")
ax.set_xlabel(variable_dist)
ax.set_ylabel("Frecuencia")
ax.legend()
st.pyplot(fig)

# 3.d Dispersión PCA
st.subheader("Visualización PCA (2 componentes)")
fig, ax = plt.subplots(figsize=(8, 6))
for cluster_id in sorted(datos["cluster"].unique()):
    subset = datos[datos["cluster"] == cluster_id]
    ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster_id}", alpha=0.6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Segmentos proyectados en componentes principales")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ============================================================
# Centroides en escala original
# ============================================================
st.header("Centroides (escala original)")
st.dataframe(centroides.round(2))
