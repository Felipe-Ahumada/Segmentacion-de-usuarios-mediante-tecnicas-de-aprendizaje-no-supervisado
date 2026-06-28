FROM python:3.12.6-slim

WORKDIR /app

# Evita archivos .pyc y fuerza salida sin buffer para ver logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD python train.py && \
    uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000
