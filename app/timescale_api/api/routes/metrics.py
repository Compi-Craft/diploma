import datetime
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..database import get_db
from ..models import MetricEntry
from ..schemas import MetricRead, SyncActualData, PredictData


router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.put("/sync")
async def sync_actual_values(data: SyncActualData, db: AsyncSession = Depends(get_db)):
    """
    [Internal] Оновлює actual_value для минулих прогнозів, час яких настав.
    Викликається Воркером.
    """
    now = datetime.datetime.utcnow()
    
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
async def save_new_prediction(data: PredictData, db: AsyncSession = Depends(get_db)):
    """
    [Internal] Зберігає новий прогноз від LSTM сервісу.
    Викликається Воркером.
    """
    target_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=data.horizon_seconds)
    
    new_entry = MetricEntry(
        resource=data.resource,
        input_value=data.input_value,
        predicted_value=data.predicted_value,
        target_ts=target_time,
        horizon_seconds=data.horizon_seconds
    )
    
    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)
    return new_entry

@router.get("/history", response_model=List[MetricRead])
async def get_history(resource: str, limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Повертає історію метрик для графіків (Дашборд / Grafana)."""
    query = (
        select(MetricEntry)
        .filter(MetricEntry.resource == resource)
        .order_by(MetricEntry.ts.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    return result.scalars().all()