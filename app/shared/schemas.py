from datetime import datetime
from typing import List, Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field


class GenericResponse(BaseModel):
    message: str
    status: str = "success"


class Health(BaseModel):
    status: str


class MetricPoint(BaseModel):
    cpu: float = Field(ge=0)
    ram: float = Field(ge=0)
    rps: float = Field(ge=0)


class PredictionRequest(BaseModel):
    history: List[MetricPoint]


class PredictionResponse(BaseModel):
    version: str
    predicted_values: MetricPoint


class StatusResponse(BaseModel):
    current_version: str
    status: str


class ReloadRequest(BaseModel):
    version: str
    model_path: str
    scaler_path: str


class RetrainCommand(BaseModel):
    target_version: str
    start_time: datetime
    end_time: datetime
    epochs: int = 5
    batch_size: int = 16


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


class MetricHistoryRead(BaseModel):
    resource: str
    limit: int = 50


class MetricHistoryRangeRead(BaseModel):
    resource: Optional[str] = None
    start_time: datetime
    end_time: datetime


class SyncActualData(BaseModel):
    resource: str
    actual_value: float


class PredictData(BaseModel):
    resource: str
    input_value: float
    predicted_value: float
    horizon_seconds: int = 60


class ModelCreate(BaseModel):
    version: Optional[str] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    model_path: str
    scaler_path: str
    is_active: bool = False


class ModelRead(ModelCreate):
    version: str
    created_at: datetime

    class Config:
        from_attributes = True


class ModelUploadRequest:
    """Клас-залежність для парсингу multipart/form-data запитів із файлами."""

    def __init__(
        self,
        version: str = Form(...),
        mse: Optional[float] = Form(None),
        mae: Optional[float] = Form(None),
        model_file: UploadFile = File(...),
        scaler_file: UploadFile = File(...),
    ):
        self.version = version
        self.mse = mse
        self.mae = mae
        self.model_file = model_file
        self.scaler_file = scaler_file


class SettingsUpdate(BaseModel):
    is_collector_active: bool
    prometheus_url: str
    cpu_query: str
    ram_query: str
    rps_query: str


class SettingsRead(SettingsUpdate):
    id: int

    class Config:
        from_attributes = True


class LogCreate(BaseModel):
    level: str
    service: str
    message: str


class LogRead(LogCreate):
    id: int
    ts: datetime

    class Config:
        from_attributes = True


class LogServiceRead(BaseModel):
    service: Optional[str] = None
    limit: int = 100
    level: Optional[str] = None
