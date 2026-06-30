"""Dashboard de segmentación de usuarios.

Consume la API del servicio ML y presenta los segmentos en tres vistas
adaptadas a la audiencia: ejecutiva, técnica y operativa.
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Segmentación de Usuarios Streaming", layout="wide")

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "font.size": 10,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
    "axes.edgecolor": "#D1D5DB",
})

st.title("Segmentación de Usuarios — Plataforma de Streaming")
st.caption("Análisis de segmentos de clientes de una plataforma de streaming mediante KMeans.")
st.divider()

@st.cache_data
def cargar_datos():
    """Descarga los datos del servicio ML. Cacheado para no repetir la petición en cada interacción."""
    respuesta = requests.get("http://ml-service:8000/dashboard-data", timeout=30)
    respuesta.raise_for_status()
    payload = respuesta.json()
    usuarios = pd.DataFrame(payload["usuarios"])
    centroides = pd.DataFrame(payload["centroides"])
    return usuarios, centroides, payload["metricas"]


try:
    usuarios, centroides, metricas = cargar_datos()
except requests.exceptions.RequestException as e:
    st.error(f"No se pudo obtener datos del servicio ML: {e}")
    st.stop()

variables_perfil = [
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km",
]

NOMBRES_SEGMENTOS = {
    0: "Usuarios frecuentes",
    1: "Ocasionales sensibles a promociones",
    2: "Usuarios intensivos premium",
}


def nombre_segmento(cluster_id):
    """Devuelve el nombre de negocio del segmento, o uno genérico si no está definido."""
    return NOMBRES_SEGMENTOS.get(cluster_id, f"Segmento {cluster_id}")


st.sidebar.header("Audiencia")
audiencia = st.sidebar.radio(
    "Vista del dashboard",
    ["Ejecutiva", "Técnica", "Operativa"],
    help="Cada vista adapta el nivel de detalle al perfil de quien analiza los resultados.",
)

st.sidebar.header("Filtros")
st.sidebar.markdown("**Segmentos a mostrar**")
clusters_disponibles = sorted(usuarios["cluster"].unique().tolist())
clusters_sel = [
    c for c in clusters_disponibles
    if st.sidebar.checkbox(nombre_segmento(c), value=True, key=f"segmento_{c}")
]

if not clusters_sel:
    st.warning("Marca al menos un segmento en la barra lateral.")
    st.stop()

datos = usuarios[usuarios["cluster"].isin(clusters_sel)]

perfil_segmentos = datos.groupby("cluster")[variables_perfil].mean().round(2)

# Promedio global, no depende de los filtros.
promedios_global = usuarios[variables_perfil].mean()


ACCIONES_SEGMENTOS = {
    0: "Recomendaciones personalizadas y fidelización para sostener su actividad.",
    1: "Campañas de retención y onboarding para aumentar el enganche temprano.",
    2: "Beneficios exclusivos y acceso anticipado para preservar su lealtad.",
}
DESCRIPCIONES_SEGMENTOS = {
    0: (
        "Es el segmento más activo en frecuencia: se conecta muchas más veces por semana y "
        "consume bastantes más contenidos que el resto, aunque en sesiones más bien cortas. "
        "Su gasto y antigüedad se mantienen cerca del promedio general, por lo que su valor "
        "está en el hábito de uso constante más que en el ticket. Constituyen una base estable "
        "sobre la cual incentivar un mayor gasto."
    ),
    1: (
        "Reúne a los clientes más nuevos y de menor actividad: bajo gasto, pocas sesiones y "
        "contenidos, y sesiones breves. Es además el grupo que más depende de promociones para "
        "consumir. Esa combinación de baja vinculación y reciente incorporación lo convierte en "
        "el segmento con mayor riesgo de abandono, donde la retención y el enganche temprano "
        "son prioritarios."
    ),
    2: (
        "Es el segmento de mayor valor: gasta bastante más que el promedio, realiza las sesiones "
        "más largas y consume la mayor variedad de géneros. Son también los clientes más antiguos "
        "y casi no dependen de promociones, lo que refleja una fidelidad consolidada. Constituyen "
        "el núcleo rentable de la plataforma y el foco debe ser preservar su lealtad."
    ),
}
PALETA = [
    "#5B5BD6", "#E8833A", "#2BB3A3", "#D6455B", "#8C6BD6",
    "#5BA0D6", "#C9A227", "#6B7280", "#B05BD6", "#3AA0A0",
]


def color_segmento(cluster_id):
    """Color fijo por segmento, para que coincida en todos los gráficos."""
    return PALETA[cluster_id % len(PALETA)]


KPIS_EJECUTIVOS = [
    "gasto_mensual",
    "horas_consumo_mensual",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "antiguedad_cliente_meses",
    "porcentaje_uso_promociones",
]
ETIQUETAS_VAR = {
    "horas_consumo_mensual": "Horas de consumo (mes)",
    "gasto_mensual": "Gasto mensual",
    "cantidad_contenidos_vistos": "Contenidos vistos (mes)",
    "sesiones_semana": "Sesiones por semana",
    "porcentaje_finalizacion": "Finalización",
    "tiempo_promedio_sesion_min": "Duración de sesión",
    "cantidad_generos_consumidos": "Géneros consumidos",
    "porcentaje_uso_promociones": "Uso de promociones",
    "antiguedad_cliente_meses": "Antigüedad",
    "edad": "Edad",
    "dispositivos_registrados": "Dispositivos registrados",
    "porcentaje_uso_app_movil": "Uso de app móvil",
    "cantidad_perfiles_creados": "Perfiles creados",
    "interacciones_mensuales_soporte": "Interacciones soporte (mes)",
    "distancia_promedio_red_km": "Distancia de red",
}


PORCENTAJE_FRACCION = ("porcentaje_uso_promociones", "porcentaje_uso_app_movil")

UNIDADES = {
    "horas_consumo_mensual": "h",
    "tiempo_promedio_sesion_min": "min",
    "antiguedad_cliente_meses": "meses",
    "edad": "años",
    "distancia_promedio_red_km": "km",
}

# Variables que en el simulador se ingresan sin decimales (enteras en los datos de origen).
VARIABLES_ENTERAS = (
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "antiguedad_cliente_meses",
    "edad",
    "dispositivos_registrados",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
)


def formato_valor(variable, valor):
    """Formatea un valor según su tipo: moneda, porcentaje, unidad o número."""
    if variable in PORCENTAJE_FRACCION:
        return f"{valor * 100:.0f}%"
    if variable == "porcentaje_finalizacion":
        return f"{valor:.0f}%"
    if variable == "gasto_mensual":
        return f"${valor:,.0f}"
    unidad = UNIDADES.get(variable)
    return f"{valor:,.1f} {unidad}" if unidad else f"{valor:,.1f}"


def formato_delta(variable, delta):
    """Formatea la diferencia respecto al promedio global, con signo y unidad.

    En variables porcentuales la diferencia se expresa en puntos porcentuales
    (p.p.) y no en %, para no confundirla con una variación relativa.
    """
    signo = "+" if delta >= 0 else "-"
    if variable in PORCENTAJE_FRACCION:
        return f"{signo}{abs(delta) * 100:.0f} p.p."
    if variable == "porcentaje_finalizacion":
        return f"{signo}{abs(delta):.0f} p.p."
    if variable == "gasto_mensual":
        return f"{signo}${abs(delta):,.0f}"
    unidad = UNIDADES.get(variable)
    return f"{signo}{abs(delta):,.1f} {unidad}" if unidad else f"{signo}{abs(delta):,.1f}"


def etiqueta_formulario(variable):
    """Etiqueta del simulador con la unidad esperada (%, $ o la unidad física)."""
    base = ETIQUETAS_VAR.get(variable, variable)
    if variable in PORCENTAJE_FRACCION or variable == "porcentaje_finalizacion":
        return f"{base} (%)"
    if variable == "gasto_mensual":
        return f"{base} ($)"
    unidad = UNIDADES.get(variable)
    if unidad and not base.endswith(")"):
        return f"{base} ({unidad})"
    return base


def desviacion_global(fila, promedios):
    """Desviación porcentual de cada variable del segmento respecto al promedio global."""
    return (fila[variables_perfil] - promedios[variables_perfil]) / promedios[variables_perfil] * 100


def clasificar_valor(fila, promedios):
    """Clasifica el segmento por valor de negocio según gasto y antigüedad."""
    gasto_alto = fila["gasto_mensual"] >= promedios["gasto_mensual"]
    fiel = fila["antiguedad_cliente_meses"] >= promedios["antiguedad_cliente_meses"]
    if gasto_alto and fiel:
        return "Segmento de alto valor y fidelizado"
    if gasto_alto and not fiel:
        return "Segmento de alto valor pero reciente"
    if not gasto_alto and fiel:
        return "Segmento de valor moderado y estable"
    return "Segmento de bajo valor y en riesgo de fuga"


# Vista ejecutiva
if audiencia == "Ejecutiva":
    st.caption("Vista orientada a negocio: tamaño de los segmentos y su interpretación estratégica.")

    conteo = datos["cluster"].value_counts().sort_index()
    porcentaje = (conteo / conteo.sum() * 100).round(1)
    orden_cl = list(perfil_segmentos.index)
    nombres = [nombre_segmento(c) for c in conteo.index]
    colores = [color_segmento(c) for c in conteo.index]

    col1, col2, col3 = st.columns(3)
    col1.metric("Segmentos identificados", metricas["k_optimo"])
    col2.metric("Usuarios analizados", f"{len(datos):,}")
    col3.metric("Gasto mensual promedio", formato_valor("gasto_mensual", datos["gasto_mensual"].mean()))

    st.divider()
    st.header("Distribución de usuarios por segmento")
    col_chart, col_tabla = st.columns([3, 2])
    with col_chart:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        barras = ax.barh(nombres, conteo.values, color=colores)
        for rect, c, p in zip(barras, conteo.values, porcentaje.values):
            ax.text(
                rect.get_width(), rect.get_y() + rect.get_height() / 2,
                f"  {c} ({p:.0f}%)", va="center", fontsize=9,
            )
        ax.set_xlabel("Usuarios")
        ax.invert_yaxis()
        ax.margins(x=0.18)
        st.pyplot(fig)
    with col_tabla:
        resumen = pd.DataFrame({
            "Segmento": [nombre_segmento(c) for c in orden_cl],
            "Usuarios": [int(conteo.get(c, 0)) for c in orden_cl],
            "Perfil de valor": [clasificar_valor(perfil_segmentos.loc[c], promedios_global) for c in orden_cl],
        })
        st.dataframe(resumen, hide_index=True, use_container_width=True)

    st.divider()
    st.header("Interpretación de negocio por segmento")
    st.caption(
        "Se priorizan los indicadores accionables por el negocio y con mayor poder discriminante "
        "entre segmentos; las demás variables del modelo (como la edad) presentan baja varianza "
        "entre grupos y se consultan en el perfil completo de la vista Operativa. En cada tarjeta, "
        "el número grande es el promedio del segmento y la flecha indica su diferencia respecto al "
        "promedio global (verde: por encima, rojo: por debajo). Las diferencias entre porcentajes "
        "se expresan en puntos porcentuales (p.p.)."
    )
    for cluster_id in perfil_segmentos.index:
        fila = perfil_segmentos.loc[cluster_id]
        n = int(conteo.get(cluster_id, 0))
        pct = porcentaje.get(cluster_id, 0)
        clasificacion = clasificar_valor(fila, promedios_global)
        with st.container(border=True):
            st.markdown(f"### {nombre_segmento(cluster_id)}")
            st.markdown(f"**{clasificacion}**  ·  {n} usuarios ({pct}% del total)")
            cols = st.columns(len(KPIS_EJECUTIVOS))
            for c, var in zip(cols, KPIS_EJECUTIVOS):
                valor = fila[var]
                # El delta compara contra el promedio de todos los usuarios, no contra la selección.
                delta = valor - promedios_global[var]
                c.metric(
                    ETIQUETAS_VAR.get(var, var),
                    formato_valor(var, valor),
                    formato_delta(var, delta),
                )
            st.markdown(DESCRIPCIONES_SEGMENTOS.get(cluster_id, ""))
            st.info(f"**Acción sugerida:** {ACCIONES_SEGMENTOS.get(cluster_id, 'Sin acción definida.')}")

            with st.expander("Ver comparación con el promedio general"):
                desv = desviacion_global(fila, promedios_global).sort_values()
                etiquetas = [ETIQUETAS_VAR.get(v, v) for v in desv.index]
                colores_barras = ["#2BB3A3" if v >= 0 else "#D6455B" for v in desv.values]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(etiquetas, desv.values, color=colores_barras)
                ax.axvline(0, color="#1A1A2E", linewidth=0.8)
                ax.set_xlabel("Diferencia relativa vs. promedio global (%)")
                ax.tick_params(labelsize=7)
                st.pyplot(fig, use_container_width=False)

# Vista técnica
elif audiencia == "Técnica":
    st.caption("Vista orientada al modelo: métricas de validación, selección de k y reducción de dimensionalidad.")
    st.info("Las métricas del modelo (k óptimo, Silhouette, varianza PCA, método del codo) corresponden al modelo entrenado completo y no se ven afectadas por los filtros.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("k óptimo", metricas["k_optimo"])
    col2.metric("Usuarios", metricas["n_usuarios"])
    col3.metric("Silhouette Score", f"{metricas['silhouette_score']:.3f}")
    col4.metric("Varianza PCA (2D)", f"{metricas['varianza_pca']:.1%}")

    st.markdown(
        "El modelo aplica **KMeans** sobre las 15 variables, previamente estandarizadas con "
        "StandardScaler para que todas pesen por igual. La cantidad de segmentos se eligió "
        "combinando dos criterios complementarios: el método del codo y el coeficiente Silhouette."
    )

    st.header("Selección del número de clusters")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Método del codo")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(metricas["rango_k"], metricas["inertias"], marker="o")
        ax.axvline(metricas["k_optimo"], color="red", linestyle="--", label=f"k = {metricas['k_optimo']}")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel("Inercia")
        ax.legend()
        st.pyplot(fig)
    with col_b:
        st.subheader("Coeficiente Silhouette")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(metricas["rango_k"], metricas["silhouettes"], marker="o", color="green")
        ax.axvline(metricas["k_optimo"], color="red", linestyle="--", label=f"k = {metricas['k_optimo']}")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel("Silhouette")
        ax.legend()
        st.pyplot(fig)

    st.markdown(
        f"Ambos criterios coinciden en **k = {metricas['k_optimo']}**. El método del codo marca el punto "
        "donde sumar más clusters deja de reducir significativamente la inercia (la dispersión interna "
        f"de los grupos), y el coeficiente Silhouette alcanza su valor más alto ({metricas['silhouette_score']:.2f}) "
        "en ese mismo número de segmentos, lo que respalda la elección."
    )

    st.header("Visualización de los segmentos")
    col_pca, col_radar = st.columns(2)
    with col_pca:
        st.subheader("Proyección PCA (2 componentes)")
        fig, ax = plt.subplots(figsize=(5, 4))
        for cluster_id in sorted(datos["cluster"].unique()):
            subset = datos[datos["cluster"] == cluster_id]
            ax.scatter(
                subset["pc1"], subset["pc2"],
                label=nombre_segmento(cluster_id),
                color=color_segmento(cluster_id),
                alpha=0.6,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    # Normalización min-max: requiere al menos dos segmentos.
    comparable = len(perfil_segmentos) >= 2
    if comparable:
        rango = perfil_segmentos.max() - perfil_segmentos.min()
        normalizado = (perfil_segmentos - perfil_segmentos.min()).div(rango.replace(0, 1))
        normalizado.index = [nombre_segmento(c) for c in perfil_segmentos.index]

    with col_radar:
        st.subheader("Perfil comparativo (radar)")
        if not comparable:
            st.info("Selecciona al menos dos segmentos para comparar perfiles.")
        else:
            categorias = [ETIQUETAS_VAR.get(v, v) for v in variables_perfil]
            N = len(categorias)
            angulos = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angulos += angulos[:1]
            fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
            for cluster_id in perfil_segmentos.index:
                valores = normalizado.loc[nombre_segmento(cluster_id)].tolist()
                valores += valores[:1]
                ax.plot(angulos, valores, label=nombre_segmento(cluster_id), color=color_segmento(cluster_id))
                ax.fill(angulos, valores, alpha=0.1, color=color_segmento(cluster_id))
            ax.set_xticks(angulos[:-1])
            ax.set_xticklabels(categorias, fontsize=6)
            ax.set_yticklabels([])
            ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=7)
            st.pyplot(fig)

    st.markdown(
        f"La proyección PCA resume las 15 variables en 2 dimensiones conservando el "
        f"**{metricas['varianza_pca']:.0%} de la varianza**; sirve para visualizar los grupos, aunque no "
        "captura toda la información. El radar compara el perfil de cada segmento en escala normalizada "
        f"(0 = el valor más bajo entre segmentos, 1 = el más alto). Un Silhouette de "
        f"**{metricas['silhouette_score']:.2f}** indica que los segmentos existen pero con solapamiento "
        "moderado: las fronteras no son nítidas, algo habitual en datos de comportamiento de usuarios."
    )

    st.subheader("Mapa de calor — promedios normalizados por segmento")
    if not comparable:
        st.info("Selecciona al menos dos segmentos para ver el mapa de calor.")
    else:
        heat = normalizado.rename(columns=ETIQUETAS_VAR)
        fig, ax = plt.subplots(figsize=(9, max(2.2, 0.5 * len(heat))))
        sns.heatmap(heat, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Segmento")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        st.pyplot(fig)
        st.caption(
            "Cada celda indica, para una variable, qué tan alto puntúa el segmento respecto a los demás "
            "(1 = el más alto, 0 = el más bajo). Permite identificar de un vistazo qué define a cada grupo."
        )

# Vista operativa
else:
    st.caption("Vista orientada a la operación: exploración detallada de cada segmento variable a variable.")

    st.header("Perfilamiento de segmentos")
    perfil_display = pd.DataFrame(index=[nombre_segmento(c) for c in perfil_segmentos.index])
    for var in variables_perfil:
        perfil_display[ETIQUETAS_VAR.get(var, var)] = [formato_valor(var, v) for v in perfil_segmentos[var]]
    st.dataframe(perfil_display, use_container_width=True)
    st.download_button(
        "Descargar perfil de segmentos (CSV)",
        data=perfil_display.to_csv().encode("utf-8"),
        file_name="perfil_segmentos.csv",
        mime="text/csv",
    )

    st.header("Distribución por variable")
    variable_dist = st.selectbox(
        "Variable a comparar",
        variables_perfil,
        format_func=lambda v: ETIQUETAS_VAR.get(v, v),
    )
    # Boxplot por segmento: compara las distribuciones sin el solapamiento de histogramas superpuestos.
    clusters_orden = sorted(datos["cluster"].unique())
    series = [datos[datos["cluster"] == c][variable_dist].dropna() for c in clusters_orden]
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(clusters_orden) + 1.5))
    cajas = ax.boxplot(
        series,
        vert=False,
        patch_artist=True,
        medianprops=dict(color="#1A1A2E", linewidth=1.5),
        flierprops=dict(
            marker="o", markersize=4, markerfacecolor="#9CA3AF",
            markeredgecolor="none", alpha=0.5,
        ),
    )
    for parche, c in zip(cajas["boxes"], clusters_orden):
        parche.set_facecolor(color_segmento(c))
        parche.set_alpha(0.75)
    ax.set_yticks(range(1, len(clusters_orden) + 1))
    ax.set_yticklabels([nombre_segmento(c) for c in clusters_orden])
    ax.set_xlabel(ETIQUETAS_VAR.get(variable_dist, variable_dist))
    ax.invert_yaxis()
    st.pyplot(fig)
    st.caption(
        "Cada caja resume la distribución del segmento: la línea central es la mediana, "
        "la caja abarca el 50% central de los usuarios y los bigotes el resto del rango; "
        "los puntos son valores atípicos."
    )

    st.header("Usuarios segmentados")
    st.caption("Listado de usuarios según los filtros aplicados.")
    st.dataframe(datos)
    st.download_button(
        "Descargar usuarios filtrados (CSV)",
        data=datos.to_csv(index=False).encode("utf-8"),
        file_name="usuarios_filtrados.csv",
        mime="text/csv",
    )

    st.header("Centroides (escala original)")
    st.caption("Los centroides provienen del modelo completo y no dependen de los filtros.")
    st.dataframe(centroides.round(2))

    st.divider()
    st.header("Simulador de clasificación")
    st.caption(
        "Ingresa las características de un usuario nuevo y el modelo lo asignará al segmento "
        "más cercano, usando el mismo KMeans entrenado (sin reentrenar)."
    )

    with st.form("simulador_usuario"):
        columnas_form = st.columns(3)
        usuario_nuevo = {}
        for i, var in enumerate(variables_perfil):
            col = columnas_form[i % 3]
            etiqueta = etiqueta_formulario(var)
            if var in PORCENTAJE_FRACCION:
                # Se ingresa en % (0-100); la conversión a la escala 0-1 del modelo la hace la API.
                usuario_nuevo[var] = col.number_input(
                    etiqueta, min_value=0.0, max_value=100.0,
                    value=float(round(promedios_global[var] * 100)), step=1.0,
                )
            elif var == "porcentaje_finalizacion":
                usuario_nuevo[var] = col.number_input(
                    etiqueta, min_value=0.0, max_value=100.0,
                    value=float(round(promedios_global[var])), step=1.0,
                )
            elif var in VARIABLES_ENTERAS:
                usuario_nuevo[var] = col.number_input(
                    etiqueta, min_value=0, value=int(round(promedios_global[var])), step=1,
                )
            else:
                usuario_nuevo[var] = col.number_input(
                    etiqueta, min_value=0.0, value=float(round(promedios_global[var], 1)),
                )
        enviado = st.form_submit_button("Clasificar usuario")

    if enviado:
        try:
            respuesta = requests.post(
                "http://ml-service:8000/predict", json=usuario_nuevo, timeout=30
            )
            respuesta.raise_for_status()
            cluster_pred = respuesta.json()["cluster"]
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo clasificar el usuario: {e}")
        else:
            st.success(f"El usuario pertenece al segmento: **{nombre_segmento(cluster_pred)}**")
            descripcion = DESCRIPCIONES_SEGMENTOS.get(cluster_pred)
            if descripcion:
                st.markdown(descripcion)
            st.info(f"**Acción sugerida:** {ACCIONES_SEGMENTOS.get(cluster_pred, 'Sin acción definida.')}")
