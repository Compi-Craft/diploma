from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from api.routes import router  # type: ignore[attr-defined]
from config import API_URL
from core.config import settings
from fastapi import FastAPI
from lstm_module import PORT
from services.model_manager import model_manager
from shared.logger import send_system_log
from shared.schemas import ModelRead
from shared.utils import async_http_request


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await send_system_log(
        "🔄 Холодний старт: перевіряємо наявність активної моделі...",
        level="INFO",
        service="lstm_module",
    )
    data = await async_http_request(
        method="GET", url=f"{API_URL}/models/active", response_model=ModelRead
    )
    if data is None:
        await send_system_log(
            "⚠️ Активну модель не знайдено в БД. Працюємо на dummy-моделі.",
            level="WARNING",
            service="lstm_module",
        )
    else:
        await send_system_log(
            f"📥 Знайдено активну модель: {data.version}. Завантажуємо...",
            level="INFO",
            service="lstm_module",
        )
        model_manager.load_new_model(
            model_path=data.model_path,
            scaler_path=data.scaler_path,
            version=data.version,
        )
    yield
    await send_system_log(
        "🛑 Зупинка сервісу Предиктора. Очищуємо ресурси...",
        level="INFO",
        service="lstm_module",
    )


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Підключаємо наші маршрути
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
