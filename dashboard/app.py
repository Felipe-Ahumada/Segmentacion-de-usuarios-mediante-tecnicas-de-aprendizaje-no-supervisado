import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Segmentación de Usuarios Streaming", layout="wide")
st.title("Segmentación de Usuarios — Plataforma de Streaming")

# Datos desde el servicio ML
try:
    respuesta = requests.get("http://ml-service:8000/dashboard-data", timeout=30)
    respuesta.raise_for_status()
    payload = respuesta.json()
except requests.exceptions.RequestException as e:
    st.error(f"No se pudo obtener datos del servicio ML: {e}")
    st.stop()

usuarios = pd.DataFrame(payload["usuarios"])
centroides = pd.DataFrame(payload["centroides"])
metricas = payload["metricas"]

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

# ============================================================
# Configuración de audiencia y filtros
# ============================================================
st.sidebar.header("Audiencia")
audiencia = st.sidebar.radio(
    "Vista del dashboard",
    ["Ejecutiva", "Técnica", "Operativa"],
    help="Cada vista adapta el nivel de detalle al perfil de quien analiza los resultados.",
)

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

if datos.empty:
    st.warning("No hay usuarios que cumplan los filtros seleccionados.")
    st.stop()

perfil_segmentos = datos.groupby("cluster")[variables_perfil].mean().round(2)


def nivel(valor, promedio_global):
    """Clasifica un valor respecto al promedio global en lenguaje de negocio."""
    if valor >= promedio_global * 1.1:
        return "alto"
    if valor <= promedio_global * 0.9:
        return "bajo"
    return "medio"


def interpretar_segmento(cluster_id, fila, promedios):
    """Genera una descripción de negocio para un segmento."""
    consumo = nivel(fila["horas_consumo_mensual"], promedios["horas_consumo_mensual"])
    gasto = nivel(fila["gasto_mensual"], promedios["gasto_mensual"])
    antiguedad = nivel(fila["antiguedad_cliente_meses"], promedios["antiguedad_cliente_meses"])
    promo = nivel(fila["porcentaje_uso_promociones"], promedios["porcentaje_uso_promociones"])
    return (
        f"Usuarios con consumo **{consumo}**, gasto mensual **{gasto}**, "
        f"antigüedad **{antiguedad}** y sensibilidad a promociones **{promo}**."
    )


# ============================================================
# VISTA EJECUTIVA
# ============================================================
if audiencia == "Ejecutiva":
    st.caption("Vista orientada a negocio: tamaño de los segmentos y su interpretación estratégica.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Segmentos identificados", metricas["k_optimo"])
    col2.metric("Usuarios analizados", f"{len(datos):,}")
    col3.metric("Calidad de la segmentación", f"{metricas['silhouette_score']:.2f}")

    st.header("Tamaño de cada segmento")
    conteo = datos["cluster"].value_counts().sort_index()
    porcentaje = (conteo / conteo.sum() * 100).round(2)

    col_a, col_b = st.columns(2)
    with col_a:
        st.bar_chart(conteo)
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            porcentaje,
            labels=[f"Segmento {c}" for c in porcentaje.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        st.pyplot(fig)

    st.header("Interpretación de negocio por segmento")
    promedios_global = datos[variables_perfil].mean()
    for cluster_id in perfil_segmentos.index:
        fila = perfil_segmentos.loc[cluster_id]
        n = int(conteo.get(cluster_id, 0))
        pct = porcentaje.get(cluster_id, 0)
        with st.container(border=True):
            st.subheader(f"Segmento {cluster_id} — {n} usuarios ({pct}%)")
            st.markdown(interpretar_segmento(cluster_id, fila, promedios_global))

# ============================================================
# VISTA TÉCNICA
# ============================================================
elif audiencia == "Técnica":
    st.caption("Vista orientada al modelo: métricas de validación, selección de k y reducción de dimensionalidad.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("k óptimo", metricas["k_optimo"])
    col2.metric("Usuarios", metricas["n_usuarios"])
    col3.metric("Silhouette Score", f"{metricas['silhouette_score']:.3f}")
    col4.metric("Varianza PCA (2D)", f"{metricas['varianza_pca']:.1%}")

    st.header("Selección del número de clusters")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Método del codo")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metricas["rango_k"], metricas["inertias"], marker="o")
        ax.axvline(metricas["k_optimo"], color="red", linestyle="--", label=f"k = {metricas['k_optimo']}")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel("Inercia")
        ax.legend()
        st.pyplot(fig)
    with col_b:
        st.subheader("Coeficiente Silhouette")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metricas["rango_k"], metricas["silhouettes"], marker="o", color="green")
        ax.axvline(metricas["k_optimo"], color="red", linestyle="--", label=f"k = {metricas['k_optimo']}")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel("Silhouette")
        ax.legend()
        st.pyplot(fig)

    st.header("Proyección PCA (2 componentes)")
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id in sorted(datos["cluster"].unique()):
        subset = datos[datos["cluster"] == cluster_id]
        ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster_id}", alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.header("Mapa de calor — promedios normalizados por cluster")
    normalizado = (perfil_segmentos - perfil_segmentos.min()) / (
        perfil_segmentos.max() - perfil_segmentos.min()
    )
    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(perfil_segmentos))))
    sns.heatmap(normalizado, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Cluster")
    st.pyplot(fig)

    st.header("Gráfico radial por segmento")
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
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

# ============================================================
# VISTA OPERATIVA
# ============================================================
else:
    st.caption("Vista orientada a la operación: exploración detallada de cada segmento variable a variable.")

    st.header("Perfilamiento de segmentos")
    st.dataframe(perfil_segmentos)

    st.header("Distribución por variable")
    variable_dist = st.selectbox("Variable a comparar", variables_perfil)
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster_id in sorted(datos["cluster"].unique()):
        subset = datos[datos["cluster"] == cluster_id][variable_dist]
        ax.hist(subset, bins=15, alpha=0.5, label=f"Cluster {cluster_id}")
    ax.set_xlabel(variable_dist)
    ax.set_ylabel("Frecuencia")
    ax.legend()
    st.pyplot(fig)

    st.header("Centroides (escala original)")
    st.dataframe(centroides.round(2))
