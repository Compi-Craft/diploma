from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiohttp
import uvicorn
from api.routes import router  # type: ignore[attr-defined]
from core.config import settings
from fastapi import FastAPI
from logger.logger import send_system_log
from lstm_module import API_URL, PORT
from services.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await send_system_log(
        "🔄 Холодний старт: перевіряємо наявність активної моделі...",
        level="INFO",
        service="lstm_module",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}/models/active") as response:
                if response.status == 200:
                    data = await response.json()
                    await send_system_log(
                        f"📥 Знайдено активну модель: {data['version']}. Завантажуємо...",
                        level="INFO",
                        service="lstm_module",
                    )

                    model_manager.load_new_model(
                        model_path=data["model_path"],
                        scaler_path=data["scaler_path"],
                        version=data["version"],
                    )
                else:
                    await send_system_log(
                        "⚠️ Активну модель не знайдено в БД. Працюємо на dummy-моделі.",
                        level="WARNING",
                        service="lstm_module",
                    )
    except aiohttp.ClientError as e:
        await send_system_log(
            f"❌ Помилка зв'язку з API під час холодного старту: {e}. Працюємо на dummy-моделі.",
            level="ERROR",
            service="lstm_module",
        )
    except Exception as e:
        await send_system_log(
            f"❌ Несподівана помилка під час холодного старту: {e}.",
            level="ERROR",
            service="lstm_module",
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
