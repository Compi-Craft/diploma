from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import SystemSettings
from ..schemas import SettingsRead, SettingsUpdate

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsRead)
async def get_settings(db: AsyncSession = Depends(get_db)) -> SettingsRead:
    """Отримує поточні налаштування. Якщо їх ще немає — створює дефолтні."""
    query = select(SystemSettings).filter(SystemSettings.id == 1)
    result = await db.execute(query)
    settings = result.scalar_one_or_none()

    # "Лінива ініціалізація": якщо таблиця пуста, створюємо перший рядок з дефолтними значеннями
    if not settings:
        settings = SystemSettings(id=1)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)

    return settings


@router.put("", response_model=SettingsRead)
async def update_settings(
    settings_in: SettingsUpdate, db: AsyncSession = Depends(get_db)
) -> SettingsRead:
    """Оновлює налаштування системи (викликається з Дашборду)."""
    query = select(SystemSettings).filter(SystemSettings.id == 1)
    result = await db.execute(query)
    settings = result.scalar_one_or_none()

    if not settings:
        # Якщо чомусь рядка немає, створюємо його з переданими даними
        settings = SystemSettings(id=1, **settings_in.model_dump())
        db.add(settings)
    else:
        # Оновлюємо існуючий рядок
        for key, value in settings_in.model_dump().items():
            setattr(settings, key, value)

    await db.commit()
    await db.refresh(settings)

    return settings
