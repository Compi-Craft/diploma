import httpx
from ..config import settings

async def get_prediction(resource_type: str, current_value: float) -> float:
    """Запит до LSTM сервісу. Якщо сервіс лежить — повертаємо поточне значення."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.PREDICTOR_URL}/predict",
                json={"resource": resource_type, "value": current_value},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json().get("prediction", current_value)
    except Exception as e:
        print(f"⚠️ Predictor Error: {e}. Using fallback (current value).")
    
    return current_value # Наївний прогноз
