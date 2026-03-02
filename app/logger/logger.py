import aiohttp
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")  # Адреса твого API

async def send_system_log(message: str, level: str, service: str = "collector_worker"):
    """Відправляє системний лог до центральної БД."""
    print(message)
    url = f"{API_URL}/logs"
    payload = {
        "level": level,
        "service": service,
        "message": message
    }
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload)
    except Exception as e:
        # Якщо сам API логів впав, просто принтимо в консоль
        print(f"Failed to send log to API: {e} | Original message: {message}")


def send_system_log_sync(message: str, level: str = "INFO", service: str = "lstm_module"):
    """
    Синхронна функція для відправки логів. 
    Ідеально підходить для важких ML-функцій, щоб не конфліктувати з asyncio.
    """
    url = f"{API_URL}/logs"  # Переконайся, що URL правильний
    payload = {
        "level": level,
        "service": service,
        "message": message
    }
    try:
        # Ставимо таймаут 2 секунди, щоб логування ніколи не "підвісило" модель
        requests.post(url, json=payload, timeout=2)
    except Exception as e:
        print(f"⚠️ Не вдалося відправити лог у БД: {e} | Повідомлення: {message}")