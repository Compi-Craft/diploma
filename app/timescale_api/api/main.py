import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sqlalchemy import text
from api.database import engine, Base
from .routes import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TimescaleAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Створення таблиць та гіпертаблиць
    logger.info("🚀 LPA API: Connecting to TimescaleDB...")
    async with engine.begin() as conn:
        # Створюємо звичайні таблиці за схемою SQLAlchemy
        await conn.run_sync(Base.metadata.create_all)
        # Виконуємо специфічну команду TimescaleDB
        await conn.execute(text(
            "SELECT create_hypertable('lpa_metrics', 'ts', if_not_exists => TRUE);"
        ))
    logger.info("✅ Database is ready.")
    yield
    # SHUTDOWN: Закриваємо з'єднання
    await engine.dispose()
    logger.info("🛑 API shut down.")

app = FastAPI(
    title="LPA Agent API",
    description="Async API for Kubernetes LSTM Predictive Autoscaler",
    lifespan=lifespan
)

# Підключення модульних роутів
app.include_router(metrics.router)

@app.get("/health")
async def health():
    return {"status": "healthy", "database": "connected"}
