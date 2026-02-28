FROM python:3.10.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Змінні оточення
ENV PYTHONPATH=/app
ENV API_URL="http://timescale_api:5000"
ENV PREDICTOR_URL="http://lstm-predictor:6000"
