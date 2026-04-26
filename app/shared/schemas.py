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
    rps: float = Field(ge=0)


class PredictionRequest(BaseModel):
    history: List[MetricPoint]


class PredictionResponse(BaseModel):
    version: str
    predicted_cpu: float


class StatusResponse(BaseModel):
    current_version: str
    status: str
    window_size: int
    forecast_horizon: int


class ReloadRequest(BaseModel):
    version: str
    model_path: str
    scaler_x_path: str
    window_size: int = 10
    forecast_horizon: int = 12


class RetrainCommand(BaseModel):
    target_version: str
    start_time: datetime
    end_time: datetime
    epochs: int = 5
    batch_size: int = 16


class PredictData(BaseModel):
    input_cpu: float
    input_ram: float
    input_rps: float
    predicted_cpu: float
    horizon_seconds: int = 60


class MetricRead(BaseModel):
    id: int
    ts: datetime
    target_ts: datetime
    input_cpu: float
    input_ram: float
    input_rps: float
    predicted_cpu: float
    actual_cpu: Optional[float]
    horizon_seconds: int
    model_version: str

    model_config = ConfigDict(from_attributes=True)


class MetricHistoryRead(BaseModel):
    limit: int = 50


class MetricHistoryRangeRead(BaseModel):
    start_time: datetime
    end_time: datetime


class SyncActualData(BaseModel):
    actual_cpu: float


class RawCpuCreate(BaseModel):
    ts: datetime
    cpu_value: float


class ModelCreate(BaseModel):
    version: Optional[str] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    model_path: str
    scaler_x_path: str
    is_active: bool = False
    window_size: int = 10
    forecast_horizon: int = 12


class ModelRead(ModelCreate):
    version: str
    created_at: datetime

    class Config:
        from_attributes = True


class ModelUploadRequest:
    def __init__(
        self,
        version: str = Form(...),
        mse: Optional[float] = Form(None),
        mae: Optional[float] = Form(None),
        window_size: int = Form(10),
        forecast_horizon: int = Form(12),
        model_file: UploadFile = File(...),
        scaler_x_file: UploadFile = File(...),
    ):
        self.version = version
        self.mse = mse
        self.mae = mae
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.model_file = model_file
        self.scaler_x_file = scaler_x_file


class SettingsUpdate(BaseModel):
    is_collector_active: bool
    prometheus_url: str
    cpu_query: str
    ram_query: str
    rps_query: str
    ood_blend_threshold: float = 0.10


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
