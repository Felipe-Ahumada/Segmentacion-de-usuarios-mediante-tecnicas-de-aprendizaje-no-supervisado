# Guia de Despliegue

## Requisitos

- Docker y Docker Compose
- Git

## Inicio rapido

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Segmentacion-de-usuarios-mediante-tecnicas-de-aprendizaje-no-supervisado

# 2. Configurar variables de entorno
cp .env.example .env

# 3. Levantar los servicios
docker compose up --build
```

Esto levanta 3 contenedores:

| Servicio | Puerto | Descripcion |
|---|---|---|
| `postgres` | 5432 | Base de datos con perfil_usuario |
| `ml-service` | 8000 | API REST (entrena modelos al iniciar) |
| `dashboard` | 8501 | Dashboard Streamlit |

## Verificacion

```bash
# API activa
curl http://localhost:8000/

# Datos del dashboard
curl http://localhost:8000/dashboard-data

# Metricas supervisadas
curl http://localhost:8000/supervised-data
```

El dashboard esta disponible en `http://localhost:8501`.

## Ejecucion de tests

```bash
# Tests unitarios locales
python -m pytest tests/ -v

# Tests dentro de Docker
docker compose run --rm tests
```

## Detener servicios

```bash
docker compose down -v
```

## Variables de entorno

Definidas en `.env` (copiar de `.env.example`):

| Variable | Descripcion |
|---|---|
| `POSTGRES_USER` | Usuario de la base de datos |
| `POSTGRES_PASSWORD` | Contrasena de la base de datos |
| `POSTGRES_DB` | Nombre de la base de datos |

## CI/CD

El pipeline de GitHub Actions (`.github/workflows/ci.yml`) ejecuta automaticamente:

1. **unit-tests**: Instala dependencias y corre pytest
2. **docker-build**: Construye imagenes, levanta servicios, verifica API y ejecuta tests en contenedor

Se activa en push a `main`, `Felipe`, `Francisca` y en pull requests a `main`.

## Troubleshooting

| Problema | Solucion |
|---|---|
| `ml-service` no arranca | Verificar que `postgres` este healthy: `docker compose ps` |
| Error de conexion a PostgreSQL | Verificar variables en `.env` |
| Train tarda mucho | Normal: la optimizacion en dos etapas evalua multiples combinaciones |
| Puerto ocupado | Cambiar mapeo en `docker-compose.yml` (ej. `"8001:8000"`) |
