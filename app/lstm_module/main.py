from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiohttp
import uvicorn
from api.routes import router  # type: ignore[attr-defined]
from core.config import settings
from fastapi import FastAPI
from lstm_module import API_URL, PORT
from services.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("🔄 Холодний старт: перевіряємо наявність активної моделі...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}/models/active") as response:
                if response.status == 200:
                    data = await response.json()
                    print(
                        f"📥 Знайдено активну модель: {data['version']}. Завантажуємо..."
                    )

                    model_manager.load_new_model(
                        model_path=data["model_path"],
                        scaler_path=data["scaler_path"],
                        version=data["version"],
                    )
                else:
                    print(
                        "⚠️ Активну модель не знайдено в БД. Працюємо на dummy-моделі."
                    )
    except aiohttp.ClientError as e:
        print(
            f"❌ Помилка зв'язку з API під час холодного старту: {e}. Працюємо на dummy-моделі."
        )
    except Exception as e:
        print(f"❌ Несподівана помилка під час холодного старту: {e}.")
    yield
    print("🛑 Зупинка сервісу Предиктора. Очищуємо ресурси...")


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Підключаємо наші маршрути
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
