import datetime

from api import MODELS_DIR, SCALERS_DIR
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base
from .utils import generate_model_version


class MetricEntry(Base):
    __tablename__ = "lpa_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        primary_key=True,
    )
    target_ts = Column(DateTime(timezone=True), index=True)
    resource = Column(String)  # 'cpu', 'ram', 'rps'
    input_value = Column(Float)
    predicted_value = Column(Float)
    actual_value = Column(Float, nullable=True)
    horizon_seconds = Column(Integer, default=60)
    model_version = Column(
        String(50), ForeignKey("lpa_models.version", ondelete="SET NULL"), nullable=True
    )
    model_info = relationship("ModelRegistry", backref="metrics")


class ModelRegistry(Base):
    __tablename__ = "lpa_models"

    # ТУТ МАГІЯ: Передаємо функцію у default (БЕЗ дужок!)
    version = Column(String(50), primary_key=True, default=generate_model_version)

    created_at = Column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )

    # Метрики якості моделі
    mse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)

    # Шляхи до файлів у Docker-контейнері
    model_path = Column(String, nullable=False)
    scaler_path = Column(String, nullable=False)

    # Статуси
    is_active = Column(Boolean, default=False)


class SystemSettings(Base):
    __tablename__ = "lpa_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)

    is_collector_active = Column(Boolean, default=True)

    prometheus_url = Column(
        String, default="http://host.docker.internal:9090/api/v1/query"
    )

    cpu_query = Column(
        String,
        default='sum(rate(container_cpu_usage_seconds_total{pod=~"podinfo-.*", container!="POD"}[1m]))',
    )
    ram_query = Column(
        String,
        default='sum(container_memory_working_set_bytes{pod=~"podinfo-.*", container!="POD"}) / 1024 / 1024',
    )
    rps_query = Column(
        String, default='sum(rate(http_requests_total{pod=~"podinfo-.*"}[1m]))'
    )


class SystemLog(Base):
    __tablename__ = "lpa_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Індексуємо за часом, щоб швидко сортувати в дашборді
    ts = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )
    level = Column(String(20), nullable=False)  # 'INFO', 'WARNING', 'ERROR'
    service = Column(
        String(50), nullable=False
    )  # 'collector_worker', 'predictor', 'api'
    message = Column(String, nullable=False)
