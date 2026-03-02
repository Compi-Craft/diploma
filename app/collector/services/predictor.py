from typing import Dict, List, Optional, cast
from logger.logger import send_system_log
import httpx

from ..config import settings


async def get_prediction(history: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Відправляє вікно з 10 останніх вимірів до LSTM сервісу.
    Очікує отримати комплексний прогноз для всіх метрик.

    :param history: Список з 10 словників, наприклад: [{"cpu": 0.1, "ram": 10.0, "rps": 150.0}, ...]
    :return: Словник з прогнозами {"cpu": ..., "ram": ..., "rps": ...} або None у разі помилки.
    """
    try:
        async with httpx.AsyncClient() as client:
            # Формуємо JSON згідно з нашою Pydantic-схемою PredictionRequest
            payload = {"history": history}

            response = await client.post(
                f"{settings.PREDICTOR_URL}/predict", json=payload, timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                # Витягуємо вкладений словник predicted_values
                return cast(Optional[Dict[str, float]], data.get("predicted_values"))
            else:
                # Якщо FastAPI повернув 400 (наприклад, не вистачає точок) або 500
                await send_system_log(f"⚠️ Predictor API Error {response.status_code}: {response.text}", level="ERROR", service="collector")

    except httpx.RequestError as e:
        await send_system_log(f"⚠️ Predictor Connection Error: Неможливо підключитися до сервісу ({e})", level="ERROR", service="collector")
    except Exception as e:
        await send_system_log(f"⚠️ Predictor Unexpected Error: {e}", level="ERROR", service="collector")

    # Замість "наївного" прогнозу, тепер краще повернути None.
    # Воркер зрозуміє це і просто не буде записувати "сміття" в базу для цього циклу.
    return None
