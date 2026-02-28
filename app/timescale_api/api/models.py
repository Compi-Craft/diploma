import datetime
from sqlalchemy import Column, Float, String, DateTime, Integer, Boolean
from .database import Base

class MetricEntry(Base):
    __tablename__ = "lpa_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    # ts — обов'язково для гіпертаблиці
    ts = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, primary_key=True)
    target_ts = Column(DateTime(timezone=True), index=True)
    resource = Column(String) # 'cpu', 'ram', 'rps'
    input_value = Column(Float)
    predicted_value = Column(Float)
    actual_value = Column(Float, nullable=True)
    horizon_seconds = Column(Integer, default=60)
    model_version = Column(String, default="v0.0.0_placeholder")


class ModelRegistry(Base):
    __tablename__ = "lpa_models"
    
    # Використовуємо версію як первинний ключ (наприклад, 'v1.0-production', 'v2026-02-28-1600')
    version = Column(String(50), primary_key=True)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    
    # Метрики якості моделі
    mse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    
    # Шляхи до файлів у Docker-контейнері
    model_path = Column(String, nullable=False)
    scaler_path = Column(String, nullable=False)
    
    # Статуси
    is_active = Column(Boolean, default=False)
    is_autotune_candidate = Column(Boolean, default=True)

class SystemSettings(Base):
    __tablename__ = "lpa_settings"
    
    # Завжди будемо використовувати id=1
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Налаштування збору (Data Collector)
    is_collector_active = Column(Boolean, default=True)
    collection_interval_sec = Column(Integer, default=15)
    
    # Налаштування Prometheus
    prometheus_url = Column(String, default="http://host.docker.internal:9090/api/v1/query")
    target_endpoint_name = Column(String, default="podinfo")
    
    # Квері (запити до Prometheus)
    cpu_query = Column(String, default='sum(rate(container_cpu_usage_seconds_total{pod=~"podinfo-.*", container!="POD"}[1m]))')
    ram_query = Column(String, default='sum(container_memory_working_set_bytes{pod=~"podinfo-.*", container!="POD"}) / 1024 / 1024')
    rps_query = Column(String, default='sum(rate(http_requests_total{pod=~"podinfo-.*"}[1m]))')
