from typing import List

from pydantic import BaseModel, Field


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
    start_time: str  # ISO формат (напр. 2026-03-01T10:00:00Z)
    end_time: str
    epochs: int = 5
    batch_size: int = 16
