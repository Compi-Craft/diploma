import os
import shutil

from config import MODELS_DIR, SCALERS_DIR
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
)
from shared.logger import send_system_log
from shared.schemas import GenericResponse, ModelCreate, ModelRead, ModelUploadRequest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ..database import get_db
from ..models import MetricEntry, ModelRegistry
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
        ".keras"
    ):
        raise HTTPException(status_code=400, detail="Модель має бути формату .keras")
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


@router.post("/{version}/evaluate", response_model=ModelRead)
async def evaluate_real_performance(
    version: str, db: AsyncSession = Depends(get_db)
) -> ModelRead:
    """
    Бере історичні прогнози моделі, порівнює їх з реальними даними
    і оновлює MSE та MAE у реєстрі моделей.
    """

    # 1. Знаходимо модель
    query_model = select(ModelRegistry).where(ModelRegistry.version == version)
    result_model = await db.execute(query_model)
    model_obj = result_model.scalar_one_or_none()

    if not model_obj:
        raise HTTPException(status_code=404, detail=f"Модель {version} не знайдена")

    # 2. Рахуємо метрики засобами бази даних (найшвидший шлях)
    # Змінна для зручності: різниця між реальністю та прогнозом
    diff = MetricEntry.actual_value - MetricEntry.predicted_value

    query_metrics = select(
        # MSE = середнє від (різниця в квадраті)
        func.avg(diff * diff).label("real_mse"),
        # MAE = середнє від (абсолютна різниця)
        func.avg(func.abs(diff)).label("real_mae"),
    ).where(
        MetricEntry.model_version == version,
        # Обов'язково беремо тільки ті записи, де реальність вже настала!
        MetricEntry.actual_value.is_not(None),
        MetricEntry.predicted_value.is_not(None),
    )

    result_metrics = await db.execute(query_metrics)
    metrics_row = result_metrics.one()

    # Якщо для моделі ще не зібралися реальні дані (всі actual_value = None)
    if metrics_row.real_mse is None or metrics_row.real_mae is None:
        raise HTTPException(
            status_code=400,
            detail="Немає достатньо реальних даних для оцінки цієї моделі.",
        )

    # 3. Оновлюємо запис моделі новими "бойовими" метриками
    model_obj.mse = float(metrics_row.real_mse)  # type: ignore[assignment]
    model_obj.mae = float(metrics_row.real_mae)  # type: ignore[assignment]

    await db.commit()
    await db.refresh(model_obj)

    # Можемо також залогувати успіх
    await send_system_log(
        f"📊 Оновлено метрики для {version}: MSE={model_obj.mse:.4f}, MAE={model_obj.mae:.4f}",
        level="INFO",
        service="timescale_api",
    )

    return ModelRead.model_validate(model_obj)
