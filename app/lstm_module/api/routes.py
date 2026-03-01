import numpy as np
from core.config import settings
from fastapi import APIRouter, BackgroundTasks, HTTPException
from models.schemas import (
    MetricPoint,
    PredictionRequest,
    PredictionResponse,
    ReloadRequest,
    StatusResponse,
)
from services.model_manager import model_manager

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if len(request.history) != settings.MODEL_INPUT_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need exactly {settings.MODEL_INPUT_STEPS} historical points",
        )

    # Конвертуємо Pydantic об'єкти в NumPy масив: shape (1, 10, 3)
    input_data = np.array([[[p.cpu, p.ram, p.rps] for p in request.history]])

    # Викликаємо модель
    prediction = model_manager.predict(input_data)

    return PredictionResponse(
        version=model_manager.version,
        predicted_values=MetricPoint(
            cpu=float(prediction[0][0]),
            ram=float(prediction[0][1]),
            rps=float(prediction[0][2]),
        ),
    )


@router.post("/reload")
async def reload_model(request: ReloadRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(
        model_manager.load_new_model, request.model_path, request.scaler_path, request.version
    )
    return {"message": f"Reloading started for {request.version}"}


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(current_version=model_manager.version, status="active")
