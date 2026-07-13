# Segmentación de usuarios mediante técnicas de aprendizaje no supervisado

Proyecto de ciencia de datos orientado a identificar grupos de usuarios con comportamientos similares dentro de una plataforma de streaming. A partir de dos fuentes de datos internas, se construye un conjunto analítico consolidado que alimenta un modelo KMeans y modelos supervisados de clasificación y regresión. Los resultados se exponen a través de una API REST y se visualizan en un dashboard interactivo con cuatro vistas diferenciadas.

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
 PostgreSQL ────────────┤    (ETL)    + Supervisado  usuarios_seg.csv  │
 perfil_usuario         │              modelos/      centroides.csv    │
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

## Tecnologías utilizadas

| Categoría | Herramientas |
|-----------|--------------|
| Lenguaje | Python 3.12 |
| Machine Learning | scikit-learn (KMeans, PCA, Silhouette, LogReg, RF, SVM, KNN, GB) |
| Selección de k | KneeLocator (kneed) |
| Tuning | RandomizedSearchCV, GridSearchCV, scipy.stats |
| Preprocesamiento | Winsorizer, CorrelationFilter (transformadores personalizados) |
| Procesamiento de datos | pandas, NumPy |
| Base de datos | PostgreSQL 16, SQLAlchemy |
| API | FastAPI, Uvicorn |
| Visualización | Streamlit, Matplotlib, Seaborn |
| Testing | pytest (36 tests) |
| Infraestructura | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Control de versiones | Git, GitHub |

## Estructura del proyecto

```
.
├── data/
│   ├── raw/
│   │   └── usuarios_streaming.csv
│   └── processed/
│       └── usuarios_integrados.csv
├── outputs/
│   ├── usuarios_segmentados.csv
│   └── centroides.csv
├── database/
│   ├── init.sql
│   └── perfil_usuarios.csv
├── ml-service/
│   ├── train.py
│   ├── app.py
│   ├── preprocesamiento.py
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── app.py
│   ├── .streamlit/config.toml
│   ├── Dockerfile
│   └── requirements.txt
├── tests/
│   ├── test_train.py
│   ├── test_api.py
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
│   ├── api.md
│   ├── arquitectura.md
│   └── deployment.md
├── .github/
│   └── workflows/ci.yml
├── docker-compose.yml
├── .env.example
└── README.md
```

## Requisitos

- Docker
- Docker Compose

No se requiere Python local ni ninguna dependencia adicional.

## Cómo levantar el proyecto

Antes de levantar los servicios, crea el archivo de variables de entorno a partir de la plantilla. Este archivo define las credenciales de PostgreSQL y no se versiona por seguridad.

```bash
cp .env.example .env
```

Luego construye y levanta los contenedores:

```bash
docker compose up --build
```

Esto levanta los tres servicios en orden. El servicio `ml-service` ejecuta primero `train.py` (integración y entrenamiento) y luego expone la API. El dashboard queda disponible una vez que la API responde.

| Servicio | URL |
|----------|-----|
| Dashboard | http://localhost:8501 |
| API ML | http://localhost:8000 |
| PostgreSQL | localhost:5432 |

## Comandos útiles

Detener los servicios:

```bash
docker compose down
```

Detener y eliminar también los volúmenes (modelos y base de datos). Es necesario si se modifica `init.sql` o `database/perfil_usuarios.csv`, ya que PostgreSQL solo ejecuta el script de inicialización cuando el volumen está vacío:

```bash
docker compose down -v
```

Reconstruir las imágenes desde cero tras cambios en dependencias o Dockerfiles:

```bash
docker compose build --no-cache
```

Inspeccionar la base de datos directamente:

```bash
docker exec -it streaming_database psql -U admin -d streaming_usuarios
```

Una vez dentro de `psql`, listar las tablas con `\dt` y consultar los datos cargados con `SELECT * FROM perfil_usuario LIMIT 5;`.

## Pipeline de entrenamiento

`ml-service/train.py` ejecuta los siguientes pasos en orden:

1. Lee `data/raw/usuarios_streaming.csv` por bloques (chunks de 10 000)
2. Consulta la tabla `perfil_usuario` desde PostgreSQL vía SQLAlchemy, también por bloques
3. Valida el esquema de cada fuente (columnas esperadas y tipos numéricos)
4. Integra ambas fuentes por `id_cliente` con merge validado (`validate="one_to_one"`)
5. Diagnostica nulos (cuantifica y registra detalle por columna) y aplica `dropna`
6. Valida rangos de columnas porcentuales y detecta outliers por IQR
7. Optimiza tipos con `pd.to_numeric(downcast=...)` para reducir uso de memoria
8. Genera resumen estadístico agrupado por rangos de antigüedad
9. Escala las 15 variables con `StandardScaler`
10. Evalúa entre 2 y 10 clusters calculando inercia y coeficiente Silhouette
11. Selecciona el `k` óptimo con `KneeLocator` sobre la curva de inercia
12. Entrena KMeans con el `k` seleccionado y calcula distancias a centroides (broadcasting)
13. Aplica PCA de 2 componentes para visualización 2D
14. Entrena modelos supervisados de clasificación y regresión con tuning en dos etapas
15. Persiste modelos, métricas y resultados en `models/` y `outputs/`

### Preprocesamiento personalizado (preprocesamiento.py)

Se implementaron dos transformadores compatibles con `sklearn.Pipeline`:

- **Winsorizer**: recorta outliers a percentiles específicos (`limits=(0.05, 0.05)`)
- **CorrelationFilter**: elimina variables con correlación absoluta > 0.9

Ambos siguen el patrón `BaseEstimator + TransformerMixin`.

### Selección del número de clusters

Se combina el método del codo (KneeLocator sobre la curva de inercia) con el coeficiente Silhouette para validar la cohesión interna de los grupos. El `k` detectado por el codo se usa como criterio de selección principal; el Silhouette sirve como métrica de calidad del resultado.

### Modelos supervisados

Después de la segmentación no supervisada, el pipeline entrena automáticamente modelos supervisados con un pipeline robusto de preprocesamiento: `Winsorizer → StandardScaler → CorrelationFilter → modelo`.

**Clasificación** — predecir el segmento de un usuario:
- Algoritmos: Logistic Regression, Random Forest, SVM, KNN
- Tuning en dos etapas: RandomizedSearchCV (n_iter=20, exploración) → GridSearchCV (refinamiento)
- Validación cruzada: StratifiedKFold(5), scoring F1 macro
- Métricas: accuracy, precision, recall, F1 macro, matriz de confusión, importancia de variables

**Regresión** — predecir el gasto mensual:
- Algoritmos: Linear Regression, Random Forest, Gradient Boosting
- Misma estructura de pipeline y tuning en dos etapas
- Validación cruzada: KFold(5), scoring R²
- Métricas: R², MAE, RMSE, importancia de variables

La selección del target de regresión (`gasto_mensual`) se justifica estadísticamente: es la variable con mayor correlación media absoluta con el resto del conjunto, además de ser el proxy más directo del valor económico del cliente.

El mejor clasificador y el mejor regresor se persisten en `models/` junto con sus métricas en `metricas_supervisado.json`.

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
    "k_optimo": 3,
    "silhouette_score": 0.231,
    "n_usuarios": 300,
    "n_clusters": 3,
    "varianza_pca": 0.449,
    "inertias": [...],
    "silhouettes": [...],
    "rango_k": [2, 3, 4, 5, 6, 7, 8, 9, 10]
  }
}
```

### `GET /supervised-data`

Retorna métricas de los modelos supervisados (clasificación y regresión): resultados por modelo, mejor modelo, importancia de variables y datos para el scatter plot de regresión.

### `POST /predict`

Clasifica un nuevo usuario en uno de los segmentos existentes usando el modelo KMeans.

Todos los porcentajes se envían en escala **0–100** (por ejemplo, `30` = 30%).

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

**Respuesta:**
```json
{
  "cluster": 1
}
```

Devuelve `400` si falta alguna variable o si los datos enviados son inválidos.

### `POST /predict-clasificador`

Clasifica un usuario usando el mejor clasificador supervisado (en lugar de KMeans directo). Mismo formato de body que `POST /predict`.

### `POST /predict-gasto`

Predice el gasto mensual de un usuario a partir de sus 14 variables restantes.

**Body esperado:** todas las variables del perfil excepto `gasto_mensual`.

**Respuesta:**
```json
{
  "gasto_mensual_estimado": 88.56
}
```

## Dashboard

El dashboard permite explorar los segmentos obtenidos seleccionando desde la barra lateral qué segmentos mostrar. Las visualizaciones se adaptan a cuatro audiencias, también seleccionables desde la barra lateral:

- **Ejecutiva**: indicadores de alto nivel, tamaño de cada segmento e interpretación de negocio generada automáticamente para cada grupo.
- **Técnica**: métricas de validación del modelo, método del codo, coeficiente Silhouette por k, proyección PCA, mapa de calor normalizado y gráfico radial.
- **Operativa**: perfilamiento detallado por segmento, distribución por variable seleccionable, centroides en escala original y un **simulador de clasificación** que asigna un usuario nuevo a su segmento mediante el endpoint `POST /predict`.
- **Supervisado**: comparación de modelos de clasificación y regresión con sus métricas (accuracy, F1, R², MAE), matrices de confusión, importancia de variables, scatter plot real vs. predicción y un **simulador supervisado** que clasifica al usuario y estima su gasto mensual.

## Resultados de la segmentación

El modelo identificó **3 segmentos** de usuarios. La siguiente tabla resume las variables de negocio más relevantes para cada uno (valores promedio según los centroides en escala original):

| Segmento | Gasto mensual | Horas consumo | Sesiones/sem | Contenidos vistos | % finalización | Antigüedad (meses) | % uso promociones |
|----------|--------------:|--------------:|-------------:|------------------:|---------------:|-------------------:|------------------:|
| 0 | 212 | 28.9 | 15.0 | 49 | 68% | 44 | 31% |
| 1 | 80 | 35.5 | 3.5 | 10 | 38% | 15 | 52% |
| 2 | 428 | 46.5 | 4.9 | 25 | 84% | 71 | 9% |

### Interpretación de negocio

- **Segmento 0 — Usuarios frecuentes.** Registran la mayor cantidad de sesiones semanales y de contenidos vistos, con gasto y antigüedad intermedios. Son usuarios muy activos por frecuencia de uso.

  *Acción sugerida:* recomendaciones personalizadas y programas de fidelización para sostener su nivel de actividad.

- **Segmento 1 — Usuarios ocasionales sensibles a promociones.** Bajo consumo, bajo gasto, baja tasa de finalización y poca antigüedad, junto con el mayor uso de promociones. Presentan el mayor riesgo de fuga.

  *Acción sugerida:* campañas de retención y onboarding que aumenten el enganche temprano.

- **Segmento 2 — Usuarios intensivos premium.** Mayor gasto mensual, sesiones más largas, mayor tasa de finalización, más géneros consumidos, mayor antigüedad y mínima dependencia de promociones. Son los clientes de mayor valor y más fieles.

  *Acción sugerida:* beneficios exclusivos y acceso anticipado a contenidos para preservar su lealtad.

> Los valores corresponden al modelo entrenado con el dataset actual (`random_state` fijo, resultados reproducibles). Si se reentrena con datos distintos, los segmentos pueden variar.

## CI/CD

El proyecto cuenta con un pipeline de integración continua en `.github/workflows/ci.yml` que se activa en push a `main`, `Felipe` y `Francisca`, y en pull requests a `main`.

El pipeline ejecuta dos jobs:

1. **unit-tests**: instala dependencias en Python 3.12 y ejecuta `pytest tests/ -v`.
2. **docker-build**: construye las imágenes, levanta los servicios, verifica que la API responda y ejecuta los tests en contenedor.

## Testing

El proyecto incluye **36 tests** en la carpeta `tests/`, con su propio contenedor definido en `docker-compose.yml` bajo el perfil `test`. Para ejecutarlos:

```bash
docker compose run --rm tests
```

Para ejecutarlos localmente sin Docker:

```bash
pip install -r tests/requirements.txt
pip install -r ml-service/requirements.txt
pytest tests/ -v
```

### Tests del pipeline ETL (`test_train.py`)

- Validación de esquema: columnas presentes, faltantes, tipos numéricos, tipos incorrectos.
- Integración de fuentes: merge por `id_cliente`, merge sin coincidencias, eliminación de nulos.
- Escalamiento: media cero y desviación estándar uno tras `StandardScaler`.
- Modelo KMeans: asignación de clusters, reproducibilidad, inercia decreciente, Silhouette válido.
- Centroides: dimensiones correctas tras inversión a escala original.

### Tests de modelos supervisados (`test_train.py`)

- Pipeline de clasificación: accuracy razonable con RandomForest.
- GridSearchCV: selecciona hiperparámetros para clasificación y regresión.
- Feature importances: el clasificador expone importancia de variables.
- Pipeline de regresión: R² definido con GradientBoosting.
- Predicciones de regresión: valores finitos.

### Tests de transformadores (`test_train.py`)

- Winsorizer: recorta outliers, preserva valores dentro del rango.
- CorrelationFilter: elimina variables colineales, preserva independientes.
- Pipeline robusto: Winsorizer + Scaler + CorrelationFilter + modelo clasifica correctamente.
- Tuning en dos etapas: RandomizedSearchCV → GridSearchCV funciona.

### Tests de la API (`test_api.py`)

- `GET /`: respuesta 200 con mensaje de estado.
- `POST /predict`: clasificación válida, error 400, reproducibilidad, usuarios extremos.
- `GET /supervised-data`: retorna datos de clasificación y regresión.
- `POST /predict-clasificador`: cluster válido, error 400 con variables faltantes.
- `POST /predict-gasto`: estimación numérica, error 400 con variables faltantes.

## Logging

El pipeline ETL y la API utilizan el módulo `logging` de Python con formato estructurado:

```
2025-06-28 14:32:01 [INFO] usuarios_streaming.csv cargado: 300 registros
2025-06-28 14:32:02 [WARNING] Nulos detectados: 2 valores en 1 filas (0.3% del dataset).
2025-06-28 14:32:03 [ERROR] No se encontró data/raw/usuarios_streaming.csv
```

Los niveles de severidad son:
- **INFO**: progreso normal del pipeline (carga, validación, entrenamiento, optimización de memoria).
- **WARNING**: situaciones recuperables (valores nulos, codo no detectado, rangos fuera de límite).
- **ERROR**: fallos que detienen la ejecución (archivo faltante, conexión fallida).

La salida se dirige a stdout y se puede consultar con `docker compose logs`.

## Colaboración

El trabajo se organiza en ramas por integrante. Cada rama concentra los cambios del respectivo desarrollador y se integra a `main` una vez revisada. El pipeline CI/CD valida automáticamente cada push.

| Integrante | Rama |
|------------|------|
| Felipe Ahumada Silva | `Felipe` |
| Francisca Carrasco Lozano | `Francisca` |
