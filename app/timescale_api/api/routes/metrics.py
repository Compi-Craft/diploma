import datetime
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..database import get_db
from ..models import MetricEntry
from ..schemas import MetricCreate, MetricRead

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.post("/", response_model=MetricRead)
async def save_prediction(data: MetricCreate, db: AsyncSession = Depends(get_db)):
    """Зберігає новий прогноз від LSTM агента."""
    target = datetime.datetime.now() + datetime.timedelta(seconds=data.horizon_seconds)
    
    new_entry = MetricEntry(
        **data.model_dump(),
        target_ts=target
    )
    
    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)
    return new_entry

@router.get("/history", response_model=List[MetricRead])
async def get_history(resource: str, limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Повертає історію метрик для графіків або навчання."""
    query = (
        select(MetricEntry)
        .filter(MetricEntry.resource == resource)
        .order_by(MetricEntry.ts.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    return result.scalars().all()
