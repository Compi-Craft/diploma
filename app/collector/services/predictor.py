import httpx
from typing import Optional, Dict, List
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
                f"{settings.PREDICTOR_URL}/predict",
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                # Витягуємо вкладений словник predicted_values
                return data.get("predicted_values")
            else:
                # Якщо FastAPI повернув 400 (наприклад, не вистачає точок) або 500
                print(f"⚠️ Predictor API Error {response.status_code}: {response.text}")
                
    except httpx.RequestError as e:
        print(f"⚠️ Predictor Connection Error: Неможливо підключитися до сервісу ({e})")
    except Exception as e:
        print(f"⚠️ Predictor Unexpected Error: {e}")
    
    # Замість "наївного" прогнозу, тепер краще повернути None.
    # Воркер зрозуміє це і просто не буде записувати "сміття" в базу для цього циклу.
    return None
