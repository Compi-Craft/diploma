import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from api import MODELS_DIR, SCALERS_DIR
from api.database import Base, engine
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Твої імпорти моделей та двигуна бази
from .database import Base, engine
from .models import ModelRegistry
from .routes import logs, metrics, model, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TimescaleAPI")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("🚀 LPA API: Connecting to TimescaleDB...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        await conn.execute(
            text(
                "SELECT create_hypertable('lpa_metrics', 'ts', if_not_exists => TRUE);"
            )
        )
    logger.info("✅ Database schema is ready.")
    async with AsyncSession(engine) as session:
        query = select(ModelRegistry).filter(ModelRegistry.is_active == True)
        result = await session.execute(query)
        active_model = result.scalar_one_or_none()

        if not active_model:
            logger.info(
                "⚠️ No active model found. Creating default 'v0-dummy' model..."
            )
            dummy_model = ModelRegistry(
                model_path=f"{MODELS_DIR}/default.h5",
                scaler_path=f"{SCALERS_DIR}/default.pkl",
                is_active=True,
                mse=0.0,
                mae=0.0,
            )
            session.add(dummy_model)
            await session.commit()
            logger.info("✅ Default model 'v0-dummy' registered and activated.")

    logger.info("🚀 API is fully started and ready to accept requests.")
    yield
    await engine.dispose()
    logger.info("🛑 API shut down.")


app = FastAPI(
    title="LPA Agent API",
    description="Async API for Kubernetes LSTM Predictive Autoscaler",
    lifespan=lifespan,
)

# Підключення модульних роутів
app.include_router(metrics.router)
app.include_router(model.router)
app.include_router(settings.router)
app.include_router(logs.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "database": "connected"}
