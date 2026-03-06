import datetime
import uuid

from config import PREDICTOR_URL
from shared.logger import send_system_log
from shared.schemas import GenericResponse, ReloadRequest
from shared.utils import async_http_request


async def notify_predictor_to_reload(
    version: str, model_path: str, scaler_path: str
) -> None:
    """Фонова задача: надсилає POST-запит до мікросервісу Предиктора"""
    try:
        request_object = ReloadRequest(
            version=version, model_path=model_path, scaler_path=scaler_path
        )
        responce = await async_http_request(
            method="POST",
            url=f"{PREDICTOR_URL}/reload",
            payload=request_object,
            response_model=GenericResponse,
        )
        if responce:
            await send_system_log(
                f"✅ Предиктор успішно отримав команду на Hot Swap до {version}",
                level="INFO",
                service="timescale_api",
            )
        else:
            await send_system_log(
                f"❌ Предиктор повернув помилку при спробі Hot Swap до {version}: {responce.message}",
                level="ERROR",
                service="timescale_api",
            )
    except Exception as e:
        await send_system_log(
            f"❌ Помилка з'єднання з Предиктором: {e}",
            level="ERROR",
            service="timescale_api",
        )


def generate_model_version() -> str:
    """Генерує унікальну версію формату: v20260301-153022-a1b2"""
    # Беремо поточний UTC час
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    # Беремо перші 4 символи випадкового UUID
    short_hash = uuid.uuid4().hex[:4]
    return f"v{timestamp}-{short_hash}"
