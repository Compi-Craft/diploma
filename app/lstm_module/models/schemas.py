from pydantic import BaseModel
from typing import List

class MetricPoint(BaseModel):
    cpu: float
    ram: float
    rps: float

class PredictionRequest(BaseModel):
    history: List[MetricPoint]

class PredictionResponse(BaseModel):
    version: str
    predicted_values: MetricPoint

class StatusResponse(BaseModel):
    current_version: str
    status: str