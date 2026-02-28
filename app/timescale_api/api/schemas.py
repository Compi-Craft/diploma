from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class MetricBase(BaseModel):
    resource: str
    input_value: float
    predicted_value: float
    horizon_seconds: int = 60

class MetricCreate(MetricBase):
    pass

class MetricRead(MetricBase):
    id: int
    ts: datetime
    target_ts: datetime
    actual_value: Optional[float]
    model_version: str
    
    # Дозволяє Pydantic працювати з об'єктами SQLAlchemy
    model_config = ConfigDict(from_attributes=True)

class SyncActualData(BaseModel):
    resource: str
    actual_value: float

class PredictData(BaseModel):
    resource: str
    input_value: float
    predicted_value: float
    horizon_seconds: int = 60

class ModelCreate(BaseModel):
    version: str
    mse: Optional[float] = None
    mae: Optional[float] = None
    model_path: str
    scaler_path: str
    is_active: bool = False
    is_autotune_candidate: bool = True

class ModelRead(ModelCreate):
    created_at: datetime

    class Config:
        from_attributes = True

class SettingsUpdate(BaseModel):
    is_collector_active: bool
    collection_interval_sec: int
    prometheus_url: str
    target_endpoint_name: str
    cpu_query: str
    ram_query: str
    rps_query: str

class SettingsRead(SettingsUpdate):
    id: int

    class Config:
        from_attributes = True
