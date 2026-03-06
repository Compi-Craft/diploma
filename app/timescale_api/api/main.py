from contextlib import asynccontextmanager
from typing import AsyncGenerator

from api.database import Base, engine
from config import MODELS_DIR, SCALERS_DIR
from fastapi import FastAPI
from shared.logger import send_system_log
from shared.schemas import Health
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Твої імпорти моделей та двигуна бази
from .database import Base, engine
from .models import ModelRegistry
from .routes import logs, metrics, model, settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await send_system_log(
        "🚀 LPA API: Connecting to TimescaleDB...",
        level="INFO",
        service="timescale_api",
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        await conn.execute(
            text(
                "SELECT create_hypertable('lpa_metrics', 'ts', if_not_exists => TRUE);"
            )
        )
    await send_system_log(
        "✅ Database schema is ready.", level="INFO", service="timescale_api"
    )
    async with AsyncSession(engine) as session:
        query = select(ModelRegistry).filter(ModelRegistry.is_active == True)
        result = await session.execute(query)
        active_model = result.scalar_one_or_none()

        if not active_model:
            await send_system_log(
                "⚠️ No active model found. Creating default 'v0-dummy' model...",
                level="WARNING",
                service="timescale_api",
            )
            dummy_model = ModelRegistry(
                model_path=f"{MODELS_DIR}/default.keras",
                scaler_path=f"{SCALERS_DIR}/default.pkl",
                is_active=True,
                mse=0.0,
                mae=0.0,
            )
            session.add(dummy_model)
            await session.commit()
            await send_system_log(
                "✅ Default model 'v0-dummy' registered and activated.",
                level="INFO",
                service="timescale_api",
            )

    await send_system_log(
        "🚀 API is fully started and ready to accept requests.",
        level="INFO",
        service="timescale_api",
    )
    yield
    await engine.dispose()
    await send_system_log("🛑 API shut down.", level="INFO", service="timescale_api")


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


@app.get("/health", response_model=Health)
async def health() -> Health:
    return Health(status="ok")
