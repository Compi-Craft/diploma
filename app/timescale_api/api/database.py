from typing import AsyncGenerator, TypeVar

from shared.logger import send_system_log
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from timescale_api.api import DATABASE_URL

# echo=False у продакшені, щоб не засмічувати логи SQL-запитами
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)
Base = declarative_base()
ModelType = TypeVar("ModelType")


# Dependency для ін'єкції сесії в роути
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Генератор сесій бази даних з автоматичним відкатом при збоях."""
    async with AsyncSessionLocal() as session:
        try:
            # Віддаємо сесію в ендпоінт
            yield session

        except Exception as e:
            # 🚨 Якщо в БУДЬ-ЯКОМУ ендпоінті сталася помилка, код повернеться сюди!
            await send_system_log(
                f"❌ Глобальний відкат транзакції через помилку: {e}",
                level="ERROR",
                service="timescale_api",
            )
            await session.rollback()
            raise  # Прокидаємо помилку далі, щоб її зловив глобальний Exception Handler
