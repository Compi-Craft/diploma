import datetime
from typing import List, Sequence

from fastapi import APIRouter, Depends
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..database import get_db
from ..models import MetricEntry, ModelRegistry
from ..schemas import MetricRead, PredictData, SyncActualData

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.put("/sync")
async def sync_actual_values(data: SyncActualData, db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """
    [Internal] Оновлює actual_value для минулих прогнозів, час яких настав.
    Викликається Воркером.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    stmt = (
        update(MetricEntry)
        .where(MetricEntry.resource == data.resource)
        .where(MetricEntry.actual_value == None)
        .where(MetricEntry.target_ts <= now)
        .values(actual_value=data.actual_value)
    )

    await db.execute(stmt)
    await db.commit()
    return {"status": "success", "message": f"Actual values synced for {data.resource}"}


@router.post("/predict", response_model=MetricRead)
async def save_new_prediction(data: PredictData, db: AsyncSession = Depends(get_db)) -> MetricRead:
    """
    [Internal] Зберігає новий прогноз від LSTM сервісу.
    Викликається Воркером.
    """

    # 1. Дістаємо версію поточної активної моделі
    # Використовуємо .where(ModelRegistry.is_active == True)
    query = select(ModelRegistry.version).filter(ModelRegistry.is_active == True)
    result = await db.execute(query)
    active_version = result.scalar_one_or_none()

    # На випадок (дуже малоймовірний), якщо активної моделі в базі чомусь немає
    if not active_version:
        active_version = "unknown-model"

    # 2. Вираховуємо час, на який зроблено прогноз
    current_time = datetime.datetime.now(datetime.timezone.utc)
    target_time = current_time + datetime.timedelta(seconds=data.horizon_seconds)

    # 3. Зберігаємо запис із прив'язкою до версії моделі
    new_entry = MetricEntry(
        resource=data.resource,
        ts = current_time,
        input_value=data.input_value,
        predicted_value=data.predicted_value,
        target_ts=target_time,
        horizon_seconds=data.horizon_seconds,
        model_version=active_version,  # 👈 ДОДАЛИ СЮДИ!
    )

    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)

    return new_entry


@router.get("/history", response_model=List[MetricRead])
async def get_history(
    resource: str, limit: int = 50, db: AsyncSession = Depends(get_db)
) -> Sequence[MetricEntry]:
    """Повертає історію метрик для графіків (Дашборд / Grafana)."""
    query = (
        select(MetricEntry)
        .filter(MetricEntry.resource == resource)
        .order_by(MetricEntry.ts.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    return result.scalars().all()
