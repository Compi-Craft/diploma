import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Урли для зв'язку між сервісами
    API_URL: str = os.getenv("API_URL", "http://timescale_api:5000") 
    PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://host.docker.internal:9090")
    PREDICTOR_URL: str = os.getenv("PREDICTOR_URL", "http://lstm-predictor:5000")
    
    COLLECTION_INTERVAL: float = 15.0
    PREDICTION_HORIZON_S: int = 60
    
    # Користувацькі запити (PromQL)
    MONITORING_QUERIES: dict = {
        # CPU для нового додатка
        "cpu": 'sum(rate(container_cpu_usage_seconds_total{pod=~"podinfo-.*", cpu="total"}[5m]))',
        
        # RAM для нового додатка (в Мегабайтах)
        "ram": 'sum(container_memory_working_set_bytes{pod=~"podinfo-.*"}) / 1024 / 1024',
        
        # Справжній HTTP RPS! Метрика віддається самим додатком.
        "rps": 'sum(rate(http_requests_total{pod=~"podinfo-.*"}[1m]))'
    }

settings = Settings()
