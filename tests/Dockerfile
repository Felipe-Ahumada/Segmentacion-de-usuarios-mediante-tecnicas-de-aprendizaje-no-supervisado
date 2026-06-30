FROM python:3.12.6-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY tests/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml-service/ ./ml-service/
COPY tests/ ./tests/

CMD ["pytest", "tests/", "-v", "--tb=short"]
