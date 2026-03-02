from typing import Any, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import SystemLog
from ..schemas import LogCreate, LogRead

router = APIRouter(prefix="/logs", tags=["logs"])


@router.post("", response_model=LogRead)
async def create_log(log_in: LogCreate, db: AsyncSession = Depends(get_db)) -> Any:
    """Мікросервіси відправляють сюди свої події та помилки."""
    new_log = SystemLog(**log_in.model_dump())
    db.add(new_log)
    await db.commit()
    await db.refresh(new_log)
    return new_log


@router.get("", response_model=List[LogRead])
async def get_recent_logs(
    limit: int = 50,
    level: Optional[str] = None,
    service: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Дашборд забирає останні логи. Можна фільтрувати за рівнем або сервісом."""

    # 1. Базовий вибір
    query = select(SystemLog)

    # 2. Спочатку фільтруємо (WHERE)
    if level:
        query = query.filter(SystemLog.level == level)

    if service:
        query = query.filter(SystemLog.service == service)

    # 3. В кінці сортуємо та обрізаємо (ORDER BY ... LIMIT ...)
    query = query.order_by(SystemLog.ts.desc()).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/services", response_model=List[str])
async def get_log_services(db: AsyncSession = Depends(get_db)) -> Any:
    """Повертає список усіх унікальних сервісів, які є в базі логів."""
    query = select(SystemLog.service).distinct()
    result = await db.execute(query)
    return result.scalars().all()
