from typing import Sequence

from fastapi import APIRouter, Depends
from shared.schemas import LogCreate, LogRead, LogServiceRead
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import SystemLog

router = APIRouter(prefix="/logs", tags=["logs"])


@router.post("", response_model=LogRead)
async def create_log(log_in: LogCreate, db: AsyncSession = Depends(get_db)) -> LogRead:
    """Мікросервіси відправляють сюди свої події та помилки."""
    new_log = SystemLog(**log_in.model_dump())
    db.add(new_log)
    await db.commit()
    await db.refresh(new_log)
    return LogRead.model_validate(new_log)


@router.get("", response_model=list[LogRead])
async def get_recent_logs(
    log_service_read: LogServiceRead,
    db: AsyncSession = Depends(get_db),
) -> list[LogRead]:
    """Дашборд забирає останні логи. Можна фільтрувати за рівнем або сервісом."""

    # 1. Базовий вибір
    query = select(SystemLog)

    # 2. Спочатку фільтруємо (WHERE)
    if log_service_read.level:
        query = query.filter(SystemLog.level == log_service_read.level)

    if log_service_read.service:
        query = query.filter(SystemLog.service == log_service_read.service)

    # 3. В кінці сортуємо та обрізаємо (ORDER BY ... LIMIT ...)
    query = query.order_by(SystemLog.ts.desc()).limit(log_service_read.limit)

    result = await db.execute(query)
    orm_logs = result.scalars().all()  # Це список об'єктів SystemLog

    # 🌟 МАГІЯ ТУТ: Генератор списку
    return [LogRead.model_validate(log) for log in orm_logs]


@router.get("/services", response_model=list[str])
async def get_log_services(db: AsyncSession = Depends(get_db)) -> Sequence[str]:
    """Повертає список усіх унікальних сервісів, які є в базі логів."""
    query = select(SystemLog.service).distinct()
    result = await db.execute(query)
    return result.scalars().all()
