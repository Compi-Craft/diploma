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