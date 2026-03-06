import os
import shutil

from config import MODELS_DIR, SCALERS_DIR
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
)
from shared.schemas import GenericResponse, ModelCreate, ModelRead, ModelUploadRequest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import ModelRegistry
from ..utils import generate_model_version, notify_predictor_to_reload

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelRead])
async def get_all_models(db: AsyncSession = Depends(get_db)) -> list[ModelRead]:
    """Повертає реєстр усіх натренованих моделей (для Дашборду)."""
    query = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
    result = await db.execute(query)
    orm_metrics = result.scalars().all()  # Це список об'єктів MetricEntry
    return [ModelRead.model_validate(model) for model in orm_metrics]


@router.get("/byversion/{version}", response_model=ModelRead)
async def get_specific_model(
    version: str, db: AsyncSession = Depends(get_db)
) -> ModelRead:
    """Отримує метадані конкретної моделі (шляхи до файлів)."""
    query = select(ModelRegistry).filter(ModelRegistry.version == version)
    result = await db.execute(query)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Модель не знайдено")
    return ModelRead.model_validate(model)


@router.post("", response_model=ModelRead)
async def publish_model(
    model_in: ModelCreate, db: AsyncSession = Depends(get_db)
) -> ModelRead:
    """Реєстрація нової моделі після донавчання."""

    # Перевіряємо, чи існує така версія
    query = select(ModelRegistry).filter(ModelRegistry.version == model_in.version)
    result = await db.execute(query)
    existing_model = result.scalar_one_or_none()

    if existing_model:
        raise HTTPException(status_code=400, detail="Модель з такою версією вже існує")

    # Якщо нова модель одразу позначається як активна,
    # треба деактивувати всі інші
    if model_in.is_active:
        await db.execute(update(ModelRegistry).values(is_active=False))

    # Створюємо запис
    new_model = ModelRegistry(**model_in.model_dump())
    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)

    return ModelRead.model_validate(new_model)


@router.put("/{version}/activate", response_model=GenericResponse)
async def activate_model(
    version: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
) -> GenericResponse:
    """Робить вибрану модель активною (Hot Swap перемикач)."""

    # 1. Перевіряємо, чи є така модель
    query = select(ModelRegistry).filter(ModelRegistry.version == version)
    result = await db.execute(query)
    target_model = result.scalar_one_or_none()

    if not target_model:
        raise HTTPException(status_code=404, detail="Модель не знайдено")

    # 2. Скидаємо прапорець is_active у всіх моделей
    await db.execute(update(ModelRegistry).values(is_active=False))

    # 3. Робимо активною потрібну
    target_model.is_active = True  # type: ignore[assignment]
    await db.commit()

    # 4. Смикаємо мікросервіс Предиктора у фоні!
    background_tasks.add_task(
        notify_predictor_to_reload,
        str(target_model.version),
        str(target_model.model_path),
        str(target_model.scaler_path),
    )

    return GenericResponse(
        message=f"Модель {version} активована в БД. Сигнал Предиктору надіслано."
    )


@router.post("/upload", response_model=ModelRead)
async def upload_custom_model(
    form_data: ModelUploadRequest = Depends(),  # 👈 Вся магія тут!
    db: AsyncSession = Depends(get_db),
) -> ModelRead:
    """Ендпоінт для завантаження власної моделі через Дашборд"""

    # 1. Перевіряємо формати
    if not form_data.model_file.filename or not form_data.model_file.filename.endswith(
        ".h5"
    ):
        raise HTTPException(status_code=400, detail="Модель має бути формату .h5")
    if (
        not form_data.scaler_file.filename
        or not form_data.scaler_file.filename.endswith(".pkl")
    ):
        raise HTTPException(status_code=400, detail="Скейлер має бути формату .pkl")
    if not form_data.version:
        version = generate_model_version()
    else:
        version = form_data.version.strip()
    # 2. Формуємо шляхи для збереження
    model_path = os.path.join(MODELS_DIR, f"{version}_{form_data.model_file.filename}")
    scaler_path = os.path.join(
        SCALERS_DIR, f"{version}_{form_data.scaler_file.filename}"
    )

    # 3. Зберігаємо файли на диск (Docker volume)
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(form_data.model_file.file, buffer)

    with open(scaler_path, "wb") as buffer:
        shutil.copyfileobj(form_data.scaler_file.file, buffer)

    # 4. Створюємо запис у базі даних (використовуємо твою схему ModelRegistry)
    new_model = ModelRegistry(
        version=version,
        mse=form_data.mse,
        mae=form_data.mae,
        model_path=model_path,
        scaler_path=scaler_path,
        is_active=False,
    )

    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)

    return ModelRead.model_validate(new_model)


@router.get("/active", response_model=ModelRead)
async def get_active_model(db: AsyncSession = Depends(get_db)) -> ModelRead:
    """Повертає поточну активну модель для холодного старту Предиктора."""
    query = select(ModelRegistry).filter(ModelRegistry.is_active == True)
    result = await db.execute(query)
    active_model = result.scalar_one_or_none()

    if not active_model:
        raise HTTPException(status_code=404, detail="Активну модель не знайдено")

    return ModelRead.model_validate(active_model)
