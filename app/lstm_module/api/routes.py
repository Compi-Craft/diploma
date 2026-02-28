import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.schemas import PredictionRequest, PredictionResponse, StatusResponse, MetricPoint
from services.model_manager import model_manager
from core.config import settings

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if len(request.history) != settings.MODEL_INPUT_STEPS:
        raise HTTPException(
            status_code=400, 
            detail=f"Need exactly {settings.MODEL_INPUT_STEPS} historical points"
        )

    # Конвертуємо Pydantic об'єкти в NumPy масив: shape (1, 10, 3)
    input_data = np.array([[ 
        [p.cpu, p.ram, p.rps] for p in request.history 
    ]])
    
    # Викликаємо модель
    prediction = model_manager.predict(input_data)
    
    return PredictionResponse(
        version=model_manager.version,
        predicted_values=MetricPoint(
            cpu=float(prediction[0][0]),
            ram=float(prediction[0][1]),
            rps=float(prediction[0][2])
        )
    )

@router.post("/reload")
async def reload_model(version: str, model_path: str, scaler_path: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(model_manager.load_new_model, model_path, scaler_path, version)
    return {"message": f"Reloading started for {version}"}

@router.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        current_version=model_manager.version, 
        status="active"
    )
    