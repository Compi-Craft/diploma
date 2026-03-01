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
