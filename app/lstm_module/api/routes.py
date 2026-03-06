import numpy as np
from core.config import settings
from fastapi import APIRouter, BackgroundTasks, HTTPException
from services.model_manager import model_manager
from services.utils import run_finetune_pipeline
from shared.schemas import (
    GenericResponse,
    MetricPoint,
    PredictionRequest,
    PredictionResponse,
    ReloadRequest,
    RetrainCommand,
    StatusResponse,
)

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


@router.post("/reload", response_model=GenericResponse)
async def reload_model(
    request: ReloadRequest, background_tasks: BackgroundTasks
) -> GenericResponse:
    try:
        background_tasks.add_task(
            model_manager.load_new_model,
            request.model_path,
            request.scaler_path,
            request.version,
        )
        return GenericResponse(message=f"Reloading started for {request.version}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start reload: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(current_version=model_manager.version, status="active")


@router.post("/retrain")
async def trigger_retraining(
    cmd: RetrainCommand, background_tasks: BackgroundTasks
) -> GenericResponse:
    """Ендпоінт, який викликає Streamlit Дашборд."""
    background_tasks.add_task(run_finetune_pipeline, cmd)
    return GenericResponse(
        message=f"Fine-tuning process for {cmd.target_version} started in the background."
    )
