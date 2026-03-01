import datetime
import uuid

import aiohttp
from api import PREDICTOR_URL


async def notify_predictor_to_reload(version: str, model_path: str, scaler_path: str) -> None:
    """Фонова задача: надсилає POST-запит до мікросервісу Предиктора"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{PREDICTOR_URL}/reload"
            payload = {"version": version, "model_path": model_path, "scaler_path": scaler_path}

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    print(
                        f"✅ Предиктор успішно отримав команду на Hot Swap до {version}"
                    )
                else:
                    error_msg = await response.text()
                    print(
                        f"❌ Предиктор повернув помилку {response.status}: {error_msg}"
                    )
    except Exception as e:
        print(f"❌ Помилка з'єднання з Предиктором: {e}")


def generate_model_version() -> str:
    """Генерує унікальну версію формату: v20260301-153022-a1b2"""
    # Беремо поточний UTC час
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    # Беремо перші 4 символи випадкового UUID
    short_hash = uuid.uuid4().hex[:4]
    return f"v{timestamp}-{short_hash}"
