# Documentación de la API REST

Servicio FastAPI que expone los modelos de segmentación y predicción.

**Base URL:** `http://localhost:8000`

---

## Endpoints

### `GET /`

Verifica que el servicio esté activo.

**Respuesta:**
```json
{"mensaje": "Servicio ML funcionando"}
```

---

### `GET /dashboard-data`

Retorna los datos completos para el dashboard: usuarios segmentados, centroides y métricas del modelo KMeans.

**Respuesta (200):**
```json
{
  "usuarios": [{"horas_consumo_mensual": 45, "cluster": 0, "pc1": 1.2, ...}],
  "centroides": [{"horas_consumo_mensual": 38.5, ...}],
  "metricas": {
    "k_optimo": 3,
    "silhouette_score": 0.23,
    "n_usuarios": 2000,
    "varianza_pca": 0.449
  }
}
```

**Error (503):** Archivos de resultados no generados aún.

---

### `GET /supervised-data`

Retorna métricas y resultados de los modelos supervisados (clasificación y regresión).

**Respuesta (200):**
```json
{
  "clasificacion": {
    "resultados": {"Random Forest": {"accuracy": 0.99, "f1_macro": 0.99, ...}},
    "mejor_modelo": "Random Forest",
    "importancia_variables": {"horas_consumo_mensual": 0.12, ...}
  },
  "regresion": {
    "resultados": {"Gradient Boosting": {"r2": 0.78, "mae": 49, ...}},
    "mejor_modelo": "Gradient Boosting",
    "y_test": [120, 200, ...],
    "y_pred": [115, 210, ...]
  }
}
```

---

### `POST /predict`

Clasifica un usuario en un segmento usando el modelo KMeans (no supervisado).

**Body (JSON):** Las 15 variables del usuario. Porcentajes en escala 0-100.

```json
{
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
  "distancia_promedio_red_km": 12.5
}
```

**Respuesta (200):**
```json
{"cluster": 1}
```

**Error (400):** Variables faltantes o datos inválidos.

---

### `POST /predict-clasificador`

Clasifica un usuario usando el mejor modelo supervisado.

**Body:** Mismo formato que `/predict`.

**Respuesta (200):**
```json
{"cluster": 1}
```

---

### `POST /predict-gasto`

Predice el gasto mensual de un usuario usando el mejor regresor.

**Body:** Las 14 variables (todas excepto `gasto_mensual`).

**Respuesta (200):**
```json
{"gasto_mensual_estimado": 135.50}
```
