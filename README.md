# Segmentación de usuarios mediante técnicas de aprendizaje no supervisado

Proyecto de ciencia de datos orientado a identificar grupos de usuarios con comportamientos similares dentro de una plataforma de streaming. A partir de dos fuentes de datos internas, se construye un conjunto analítico consolidado que alimenta un modelo KMeans. Los resultados se exponen a través de una API REST y se visualizan en un dashboard interactivo.

## Arquitectura

El sistema está compuesto por tres servicios orquestados con Docker Compose:

- **postgres**: base de datos relacional que almacena el perfil complementario de cada usuario.
- **ml-service**: servicio Python que ejecuta el pipeline de integración, entrenamiento del modelo y exposición de resultados vía FastAPI.
- **dashboard**: aplicación Streamlit que consume la API del servicio ML y presenta los segmentos de forma visual.

```
                        ┌─────────────────────────────────────────────┐
                        │              ml-service                      │
                        │                                              │
 usuarios_streaming.csv─┤                                              │
                        │  train.py ──► KMeans ──► outputs/           │
 PostgreSQL ────────────┤    (ETL)       (modelo)   usuarios_seg.csv  │
 perfil_usuario         │                           centroides.csv    │
                        │                                              │
                        │  app.py (FastAPI :8000)                      │
                        └──────────────────┬───────────────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │  dashboard   │
                                    │  Streamlit   │
                                    │   :8501      │
                                    └─────────────┘
```

## Fuentes de datos

| Fuente | Tipo | Descripción |
|--------|------|-------------|
| `data/raw/usuarios_streaming.csv` | CSV | Hábitos de consumo: horas mensuales, gasto, sesiones, géneros, promociones, antigüedad |
| `perfil_usuario` (PostgreSQL) | SQL | Perfil del usuario: edad, dispositivos, uso móvil, perfiles creados, soporte, red |

El contenido de `database/perfil_usuarios.csv` se carga automáticamente en PostgreSQL al inicializar el contenedor mediante `database/init.sql`.

### Variables del conjunto integrado

| Variable | Fuente | Descripción |
|----------|--------|-------------|
| `horas_consumo_mensual` | streaming | Horas totales consumidas en el mes |
| `gasto_mensual` | streaming | Gasto asociado al servicio en el mes |
| `cantidad_contenidos_vistos` | streaming | Cantidad de títulos vistos |
| `sesiones_semana` | streaming | Promedio de sesiones por semana |
| `porcentaje_finalizacion` | streaming | Porcentaje de contenidos terminados |
| `tiempo_promedio_sesion_min` | streaming | Duración promedio de cada sesión en minutos |
| `cantidad_generos_consumidos` | streaming | Diversidad de géneros consumidos |
| `porcentaje_uso_promociones` | streaming | Proporción de consumo con promociones activas |
| `antiguedad_cliente_meses` | streaming | Meses desde el registro del cliente |
| `edad` | perfil | Edad del usuario |
| `dispositivos_registrados` | perfil | Cantidad de dispositivos vinculados |
| `porcentaje_uso_app_movil` | perfil | Proporción de uso desde aplicación móvil |
| `cantidad_perfiles_creados` | perfil | Perfiles creados dentro de la cuenta |
| `interacciones_mensuales_soporte` | perfil | Contactos con soporte en el mes |
| `distancia_promedio_red_km` | perfil | Distancia promedio asociada a la red de conexión |

## Estructura del proyecto

```
.
├── data/
│   ├── raw/
│   │   └── usuarios_streaming.csv       # input crudo, versionado
│   └── processed/                       # merge generado, ignorado por git
├── outputs/                             # resultados del modelo, ignorado por git
│   ├── usuarios_segmentados.csv
│   └── centroides.csv
├── database/
│   ├── init.sql                         # crea tabla y carga CSV en PostgreSQL
│   └── perfil_usuarios.csv
├── ml-service/
│   ├── train.py                         # pipeline ETL + entrenamiento
│   ├── app.py                           # API FastAPI
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── app.py                           # visualización Streamlit
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```

## Requisitos

- Docker
- Docker Compose

No se requiere Python local ni ninguna dependencia adicional.

## Cómo levantar el proyecto

```bash
docker compose up --build
```

Esto levanta los tres servicios en orden. El servicio `ml-service` ejecuta primero `train.py` (integración y entrenamiento) y luego expone la API. El dashboard queda disponible una vez que la API responde.

| Servicio | URL |
|----------|-----|
| Dashboard | http://localhost:8501 |
| API ML | http://localhost:8000 |
| PostgreSQL | localhost:5432 |

Para detener todos los servicios:

```bash
docker compose down
```

## Pipeline de entrenamiento

`ml-service/train.py` ejecuta los siguientes pasos en orden:

1. Lee `data/raw/usuarios_streaming.csv`
2. Consulta la tabla `perfil_usuario` desde PostgreSQL vía SQLAlchemy
3. Integra ambas fuentes por `id_cliente` y guarda el resultado en `data/processed/`
4. Elimina la columna `id_cliente` y escala todas las variables con `StandardScaler`
5. Evalúa entre 2 y 10 clusters calculando inercia y coeficiente Silhouette para cada `k`
6. Selecciona el `k` óptimo automáticamente usando `KneeLocator` sobre la curva de inercia
7. Entrena el modelo final con el `k` seleccionado
8. Aplica PCA de 2 componentes para permitir visualización en 2D
9. Guarda en `outputs/`: usuarios con cluster y componentes PCA, centroides en escala original
10. Persiste el modelo, scaler y PCA en el volumen `models/` para que la API los cargue

### Selección del número de clusters

Se combina el método del codo (KneeLocator sobre la curva de inercia) con el coeficiente Silhouette para validar la cohesión interna de los grupos. El `k` detectado por el codo se usa como criterio de selección principal; el Silhouette sirve como métrica de calidad del resultado.

## API

La documentación interactiva de la API está disponible en http://localhost:8000/docs una vez levantado el servicio.

### `GET /`

Verificación de estado del servicio.

**Respuesta:**
```json
{
  "mensaje": "Servicio ML funcionando"
}
```

### `GET /dashboard-data`

Retorna los usuarios segmentados, los centroides y las métricas del modelo en formato JSON. Es el endpoint que consume el dashboard.

**Respuesta:**
```json
{
  "usuarios": [...],
  "centroides": [...],
  "metricas": {
    "k_optimo": 4,
    "silhouette_score": 0.312,
    "n_usuarios": 1000,
    "n_clusters": 4,
    "varianza_pca": 0.61,
    "inertias": [...],
    "silhouettes": [...],
    "rango_k": [2, 3, 4, 5, 6, 7, 8, 9, 10]
  }
}
```

### `POST /predict`

Clasifica un nuevo usuario en uno de los segmentos existentes usando el modelo entrenado.

**Body esperado:**
```json
{
  "horas_consumo_mensual": 45,
  "gasto_mensual": 120,
  "cantidad_contenidos_vistos": 18,
  "sesiones_semana": 5,
  "porcentaje_finalizacion": 72,
  "tiempo_promedio_sesion_min": 95,
  "cantidad_generos_consumidos": 6,
  "porcentaje_uso_promociones": 0.3,
  "antiguedad_cliente_meses": 24,
  "edad": 34,
  "dispositivos_registrados": 2,
  "porcentaje_uso_app_movil": 0.65,
  "cantidad_perfiles_creados": 3,
  "interacciones_mensuales_soporte": 1,
  "distancia_promedio_red_km": 12.5
}
```

**Respuesta:**
```json
{
  "cluster": 2
}
```

## Dashboard

El dashboard permite explorar los segmentos obtenidos mediante filtros interactivos por cluster y antigüedad del cliente. Incluye las siguientes secciones:

- **Visualización general**: cantidad de usuarios por cluster y distribución porcentual
- **Perfilamiento**: promedios de cada variable por segmento
- **Comparación entre segmentos**: mapa de calor normalizado, gráfico radial y distribución por variable seleccionable
- **Proyección PCA**: dispersión de usuarios en 2 componentes principales con color por cluster
- **Centroides**: tabla de centroides en escala original para interpretación directa

## Colaboración

El trabajo se organiza en ramas por integrante. Cada rama concentra los cambios del respectivo desarrollador y se integra a `main` una vez revisada.

| Integrante | Rama |
|------------|------|
| Felipe Ahumada Silva | `Felipe` |
| Francisca Carrasco Lozano | `Francisca` |
