# Arquitectura del Sistema

## Vision general

Sistema de segmentacion de usuarios de una plataforma de streaming, que combina aprendizaje no supervisado (KMeans) con modelos supervisados (clasificacion y regresion). Desplegado con Docker Compose y CI/CD via GitHub Actions.

```
                    ┌──────────────┐
                    │  PostgreSQL  │
                    │ perfil_user  │
                    └──────┬───────┘
                           │
┌──────────────┐           │         ┌──────────────┐
│  CSV streaming├───────►  ETL  ◄────┤  Validacion  │
└──────────────┘           │         │  de esquemas │
                           │         └──────────────┘
                           ▼
                    ┌──────────────┐
                    │   KMeans     │
                    │ Segmentacion │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Supervisado  │
                    │ Clasif + Reg │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   FastAPI    │
                    │   API REST   │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │  Streamlit   │
                    │  Dashboard   │
                    └──────────────┘
```

## Componentes

### ETL (`ml-service/train.py → cargar_datos()`)

- **Fuentes**: CSV (usuarios_streaming.csv) + PostgreSQL (perfil_usuario)
- **Lectura por bloques**: `chunksize=10000` para escalar a grandes volumenes
- **Validacion**: Verifica columnas esperadas y tipos numericos
- **Integracion**: Merge por `id_cliente`, eliminacion de nulos

### Aprendizaje no supervisado (`entrenar_kmeans()`)

- **Escalado**: StandardScaler (z-score)
- **Seleccion de k**: Metodo del codo (KneeLocator) + Silhouette Score
- **Modelo**: KMeans (random_state=29, n_init=10)
- **Reduccion**: PCA (2 componentes) para visualizacion

### Aprendizaje supervisado (`entrenar_supervisado()`)

**Pipeline de preprocesamiento** (por modelo):
1. `Winsorizer` — recorta outliers por percentil (5-95)
2. `StandardScaler` — normalizacion z-score
3. `CorrelationFilter` — elimina variables con correlacion > 0.9

**Optimizacion en dos etapas**:
1. `RandomizedSearchCV` (n_iter=20) — exploracion amplia con distribuciones
2. `GridSearchCV` — refinamiento alrededor del mejor resultado

**Clasificacion** (target: cluster asignado por KMeans):
- Logistic Regression, Random Forest, SVM, KNN
- StratifiedKFold (5 folds, shuffle, random_state=29)
- Metrica: F1 macro

**Regresion** (target: gasto_mensual):
- Linear Regression, Random Forest, Gradient Boosting
- KFold (5 folds, shuffle, random_state=29)
- Metrica: R2

### API REST (`ml-service/app.py`)

- Framework: FastAPI
- 6 endpoints: salud, dashboard-data, supervised-data, predict, predict-clasificador, predict-gasto
- Carga modelos desde pickle al iniciar

### Dashboard (`dashboard/app.py`)

- Framework: Streamlit
- 4 vistas por audiencia: Ejecutiva, Tecnica, Operativa, Supervisado
- Consume datos de la API REST

## Infraestructura

### Docker Compose (3 servicios + tests)

| Servicio | Imagen | Dependencias |
|---|---|---|
| postgres | postgres:16 | - |
| ml-service | python:3.12-slim | postgres (healthy) |
| dashboard | python:3.12-slim | ml-service (healthy) |
| tests | python:3.12-slim | perfil: test |

### CI/CD (GitHub Actions)

- **unit-tests**: pytest en Python 3.12
- **docker-build**: build + healthcheck + verificacion de API + tests en contenedor
