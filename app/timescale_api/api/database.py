import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from . import DATABASE_URL

# echo=False у продакшені, щоб не засмічувати логи SQL-запитами
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

# Dependency для ін'єкції сесії в роути
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session