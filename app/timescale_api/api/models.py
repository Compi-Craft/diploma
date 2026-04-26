import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base
from .utils import generate_model_version


class MetricEntry(Base):
    __tablename__ = "lpa_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        primary_key=True,
    )
    target_ts = Column(DateTime(timezone=True), index=True)
    input_cpu = Column(Float)
    input_ram = Column(Float)
    input_rps = Column(Float)
    predicted_cpu = Column(Float)
    actual_cpu = Column(Float, nullable=True)
    horizon_seconds = Column(Integer, default=60)
    model_version = Column(
        String(50), ForeignKey("lpa_models.version", ondelete="SET NULL"), nullable=True
    )
    model_info = relationship("ModelRegistry", backref="metrics")


class ModelRegistry(Base):
    __tablename__ = "lpa_models"

    version = Column(String(50), primary_key=True, default=generate_model_version)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
    mse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    model_path = Column(String, nullable=False)
    scaler_x_path = Column(String, nullable=False)
    window_size = Column(Integer, default=10)
    forecast_horizon = Column(Integer, default=12)
    is_active = Column(Boolean, default=False)


class SystemSettings(Base):
    __tablename__ = "lpa_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    is_collector_active = Column(Boolean, default=False)
    prometheus_url = Column(String, default="http://192.168.49.2:30090/api/v1/query")
    cpu_query = Column(
        String,
        default=(
            'sum(rate(container_cpu_usage_seconds_total{pod=~"cpu-service-.*", container!="POD"}[1m]))'
            " / "
            'sum(kube_pod_container_resource_limits{pod=~"cpu-service-.*", resource="cpu", container!="POD"})'
        ),
    )
    ram_query = Column(
        String,
        default='sum(container_memory_working_set_bytes{pod=~"cpu-service-.*", container!="POD"}) / 1024 / 1024',
    )
    rps_query = Column(
        String, default='sum(rate(http_requests_total{pod=~"cpu-service-.*"}[30s]))'
    )
    ood_blend_threshold = Column(Float, default=0.10)


class RawCpuReading(Base):
    __tablename__ = "lpa_raw_cpu"

    ts = Column(DateTime(timezone=True), primary_key=True)
    cpu_value = Column(Float, nullable=False)


class SystemLog(Base):
    __tablename__ = "lpa_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )
    level = Column(String(20), nullable=False)
    service = Column(String(50), nullable=False)
    message = Column(String, nullable=False)
