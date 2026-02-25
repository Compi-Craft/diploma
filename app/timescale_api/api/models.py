import datetime
from sqlalchemy import Column, Float, String, DateTime, Integer
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