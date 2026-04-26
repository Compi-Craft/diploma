import os
import shutil

import numpy as np
from config import MODELS_DIR, SCALERS_DIR
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
)
from shared.logger import send_system_log
from shared.schemas import GenericResponse, ModelCreate, ModelRead, ModelUploadRequest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import MetricEntry, ModelRegistry
from ..utils import generate_model_version, notify_predictor_to_reload

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelRead])
async def get_all_models(db: AsyncSession = Depends(get_db)) -> list[ModelRead]:
    query = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
    result = await db.execute(query)
    orm_models = result.scalars().all()
    return [ModelRead.model_validate(model) for model in orm_models]


@router.get("/byversion/{version}", response_model=ModelRead)
async def get_specific_model(
    version: str, db: AsyncSession = Depends(get_db)
) -> ModelRead:
    query = select(ModelRegistry).filter(ModelRegistry.version == version)
    result = await db.execute(query)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelRead.model_validate(model)


@router.post("", response_model=ModelRead)
async def publish_model(
    model_in: ModelCreate, db: AsyncSession = Depends(get_db)
) -> ModelRead:
    """Register a new model after fine-tuning."""
    query = select(ModelRegistry).filter(ModelRegistry.version == model_in.version)
    result = await db.execute(query)
    existing_model = result.scalar_one_or_none()

    if existing_model:
        raise HTTPException(status_code=400, detail="Model version already exists")

    if model_in.is_active:
        await db.execute(update(ModelRegistry).values(is_active=False))

    new_model = ModelRegistry(**model_in.model_dump())
    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)

    return ModelRead.model_validate(new_model)


@router.put("/{version}/activate", response_model=GenericResponse)
async def activate_model(
    version: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
) -> GenericResponse:
    """Make a model active and signal the predictor to hot-swap."""
    query = select(ModelRegistry).filter(ModelRegistry.version == version)
    result = await db.execute(query)
    target_model = result.scalar_one_or_none()

    if not target_model:
        raise HTTPException(status_code=404, detail="Model not found")

    await db.execute(update(ModelRegistry).values(is_active=False))
    target_model.is_active = True  # type: ignore[assignment]
    await db.commit()

    background_tasks.add_task(
        notify_predictor_to_reload,
        str(target_model.version),
        str(target_model.model_path),
        str(target_model.scaler_x_path),
        int(target_model.window_size or 10),
        int(target_model.forecast_horizon or 12),
    )

    return GenericResponse(
        message=f"Model {version} activated. Reload signal sent to predictor."
    )


@router.post("/upload", response_model=ModelRead)
async def upload_custom_model(
    form_data: ModelUploadRequest = Depends(),
    db: AsyncSession = Depends(get_db),
) -> ModelRead:
    """Upload a trained model and its Scaler X via the Dashboard."""
    if not form_data.model_file.filename or not form_data.model_file.filename.endswith(
        ".keras"
    ):
        raise HTTPException(status_code=400, detail="Model must be a .keras file")

    if (
        not form_data.scaler_x_file.filename
        or not form_data.scaler_x_file.filename.endswith(".joblib")
    ):
        raise HTTPException(status_code=400, detail="Scaler X must be a .joblib file")

    version = (
        form_data.version.strip() if form_data.version else generate_model_version()
    )

    model_path = os.path.join(MODELS_DIR, f"{version}_{form_data.model_file.filename}")
    scaler_x_path = os.path.join(
        SCALERS_DIR, f"{version}_X_{form_data.scaler_x_file.filename}"
    )

    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(form_data.model_file.file, buffer)

    with open(scaler_x_path, "wb") as buffer:
        shutil.copyfileobj(form_data.scaler_x_file.file, buffer)

    new_model = ModelRegistry(
        version=version,
        mse=form_data.mse,
        mae=form_data.mae,
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        window_size=form_data.window_size,
        forecast_horizon=form_data.forecast_horizon,
        is_active=False,
    )

    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)

    return ModelRead.model_validate(new_model)


@router.get("/active", response_model=ModelRead)
async def get_active_model(db: AsyncSession = Depends(get_db)) -> ModelRead:
    """Return the currently active model — used for predictor cold start."""
    query = select(ModelRegistry).filter(ModelRegistry.is_active == True)
    result = await db.execute(query)
    active_model = result.scalar_one_or_none()

    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found")

    return ModelRead.model_validate(active_model)


@router.post("/{version}/evaluate", response_model=ModelRead)
async def evaluate_real_performance(
    version: str, db: AsyncSession = Depends(get_db)
) -> ModelRead:
    """
    Compute MSE and MAE on closed predictions (where actual_cpu is known).
    Uses CPU utilization in [0, 1].
    """
    query_model = select(ModelRegistry).where(ModelRegistry.version == version)
    result_model = await db.execute(query_model)
    model_obj = result_model.scalar_one_or_none()

    if not model_obj:
        raise HTTPException(status_code=404, detail=f"Model {version} not found")

    query_data = select(MetricEntry).where(
        MetricEntry.model_version == version,
        MetricEntry.actual_cpu.is_not(None),
        MetricEntry.predicted_cpu.is_not(None),
    )
    result_data = await db.execute(query_data)
    entries = result_data.scalars().all()

    if not entries:
        raise HTTPException(
            status_code=400,
            detail="No closed predictions available to evaluate this model.",
        )

    mse_list: list[float] = []
    mae_list: list[float] = []

    for e in entries:
        error = float(e.actual_cpu) - float(e.predicted_cpu)
        mse_list.append(error**2)
        mae_list.append(abs(error))

    model_obj.mse = float(np.mean(mse_list))  # type: ignore[assignment]
    model_obj.mae = float(np.mean(mae_list))  # type: ignore[assignment]

    await db.commit()
    await db.refresh(model_obj)

    await send_system_log(
        f"Metrics updated for {version}: MSE={model_obj.mse:.6f}, MAE={model_obj.mae:.6f}",
        level="INFO",
        service="timescale_api",
    )

    return ModelRead.model_validate(model_obj)
