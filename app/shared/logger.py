import aiohttp
import requests
from config import API_URL


async def send_system_log(message: str, level: str, service: str = "collector") -> None:
    """Send a log entry to the central DB."""
    print(message)
    url = f"{API_URL}/logs"
    payload = {"level": level, "service": service, "message": message}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send log to API: {e} | Original: {message}")


def send_system_log_sync(
    message: str, level: str = "INFO", service: str = "predictor"
) -> None:
    """Synchronous log sender for use inside heavy ML functions that run outside asyncio."""
    url = f"{API_URL}/logs"
    payload = {"level": level, "service": service, "message": message}
    try:
        requests.post(url, json=payload, timeout=2)
    except Exception as e:
        print(f"Failed to send log: {e} | Message: {message}")
