FROM python:3.12.6-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD python -c "import os; exit(0) if os.path.exists('models/modelo_kmeans.pkl') else exit(1)" \
    || python train.py && \
    uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000
