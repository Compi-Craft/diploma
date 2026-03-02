import os
import shutil
from typing import Any, List, Sequence

from api import MODELS_DIR, SCALERS_DIR
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import ModelRegistry
from ..schemas import ModelCreate, ModelRead
from ..utils import notify_predictor_to_reload

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelRead])
async def get_all_models(db: AsyncSession = Depends(get_db)) -> Sequence[ModelRegistry]:
    """Повертає реєстр усіх натренованих моделей (для Дашборду)."""
    query = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/byversion/{version}", response_model=ModelRead)
async def get_specific_model(version: str, db: AsyncSession = Depends(get_db)):
    """Отримує метадані конкретної моделі (шляхи до файлів)."""
    query = select(ModelRegistry).filter(ModelRegistry.version == version)
    result = await db.execute(query)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Модель не знайдено")
    return model

@router.post("", response_model=ModelRead)
async def publish_model(model_in: ModelCreate, db: AsyncSession = Depends(get_db)) -> ModelRead:
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

    return new_model


@router.put("/{version}/activate")
async def activate_model(
    version: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
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

    return {
        "message": f"Модель {version} активована в БД. Сигнал Предиктору надіслано.",
        "model_path": target_model.model_path,
    }


@router.post("/upload", response_model=ModelRead)
async def upload_custom_model(
    version: str = Form(...),
    mse: float = Form(None),
    mae: float = Form(None),
    model_file: UploadFile = File(...),
    scaler_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> ModelRead:
    """Ендпоінт для завантаження власної моделі через Дашборд"""

    # 1. Перевіряємо формати
    if not model_file.filename or not model_file.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="Модель має бути формату .h5")
    if not scaler_file.filename or not scaler_file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Скейлер має бути формату .pkl")
    if not version:
        from ..utils import generate_model_version  # імпорт твоєї функції

        version = generate_model_version()
    # 2. Формуємо шляхи для збереження
    model_path = os.path.join(MODELS_DIR, f"{version}_{model_file.filename}")
    scaler_path = os.path.join(SCALERS_DIR, f"{version}_{scaler_file.filename}")

    # 3. Зберігаємо файли на диск (Docker volume)
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(model_file.file, buffer)

    with open(scaler_path, "wb") as buffer:
        shutil.copyfileobj(scaler_file.file, buffer)

    # 4. Створюємо запис у базі даних (використовуємо твою схему ModelRegistry)
    new_model = ModelRegistry(
        version=version,
        mse=mse,
        mae=mae,
        model_path=model_path,
        scaler_path=scaler_path,
        is_active=False,
    )

    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)

    return new_model


@router.get("/active", response_model=ModelRead)
async def get_active_model(db: AsyncSession = Depends(get_db)) -> ModelRead:
    """Повертає поточну активну модель для холодного старту Предиктора."""
    query = select(ModelRegistry).filter(ModelRegistry.is_active == True)
    result = await db.execute(query)
    active_model = result.scalar_one_or_none()

    if not active_model:
        raise HTTPException(status_code=404, detail="Активну модель не знайдено")

    return active_model
