import asyncio
import datetime
from collections import deque
from typing import Deque, Dict

from logger.logger import send_system_log
from prometheus_client import Gauge, start_http_server

from .services import api_client, predictor, prometheus

# --- Оголошуємо наші метрики ---
PREDICTED_CPU = Gauge(
    "lstm_predicted_cpu_cores", "Predicted CPU usage in cores for the next window"
)
PREDICTED_RAM = Gauge("lstm_predicted_ram_mb", "Predicted RAM usage in MB")
PREDICTED_RPS = Gauge("lstm_predicted_rps", "Predicted Requests Per Second")
# -------------------------------------

is_busy = False
history_buffer: Deque[Dict[str, float]] = deque(maxlen=10)


async def restore_history_buffer() -> None:
    """Відновлює останні 10 точок з бази даних для швидкого старту LSTM."""
    await send_system_log(
        "🔄 Спроба відновити історію з БД для швидкого старту...",
        level="INFO",
        service="collector",
    )
    try:
        # 1. Отримуємо останні 10 записів
        cpu_data = await api_client.get_recent_history("cpu", limit=10)
        ram_data = await api_client.get_recent_history("ram", limit=10)
        rps_data = await api_client.get_recent_history("rps", limit=10)

        # 2. API зазвичай повертає найновіші першими (DESC сортування).
        # Але для LSTM критично важливий хронологічний порядок (від старого до нового).
        cpu_data.reverse()
        ram_data.reverse()
        rps_data.reverse()

        # 3. Беремо мінімальну довжину (на випадок, якщо даних у базі ще мало)
        min_len = min(len(cpu_data), len(ram_data), len(rps_data))

        for i in range(min_len):
            point = {
                "cpu": cpu_data[i].get("input_value", 0.0),
                "ram": ram_data[i].get("input_value", 0.0),
                "rps": rps_data[i].get("input_value", 0.0),
            }
            history_buffer.append(point)

        if min_len > 0:
            msg = f"✅ Буфер відновлено! Завантажено {min_len}/10 точок з БД."
            await api_client.send_system_log(msg, "INFO", "collector")
        else:
            await api_client.send_system_log(
                "ℹ️ База порожня. Буфер почне заповнюватися з нуля.",
                "INFO",
                "collector",
            )

    except Exception as e:
        await api_client.send_system_log(
            f"⚠️ Помилка відновлення буфера: {e}", "ERROR", "collector"
        )


async def process_metrics_task(sys_settings: dict) -> None:
    global is_busy
    if is_busy:
        return

    is_busy = True
    await send_system_log("🕒 Початок збору метрик", level="INFO", service="collector")

    try:
        prom_url: str = sys_settings.get("prometheus_url") or ""
        queries = {
            "cpu": sys_settings.get("cpu_query"),
            "ram": sys_settings.get("ram_query"),
            "rps": sys_settings.get("rps_query"),
        }

        current_metrics = {}
        max_retries = 3
        retry_delay = 5
        fetch_success = False

        # 1. Захищений збір реальних даних (з ретраями)
        for attempt in range(max_retries):
            all_metrics_ok = True

            for resource, query in queries.items():
                if not query:
                    continue

                try:
                    val = await prometheus.fetch_metric(query, prom_url=prom_url)
                except Exception as e:
                    val = None
                    await send_system_log(
                        f"⚠️ Помилка з'єднання з Prometheus: {e}",
                        level="ERROR",
                        service="collector",
                    )

                if val is None:
                    await send_system_log(
                        f"   ⏳ Не вдалося отримати '{resource}'. Спроба {attempt + 1} з {max_retries}. Чекаємо {retry_delay} сек...",
                        level="WARNING",
                        service="collector",
                    )
                    all_metrics_ok = False
                    break  # Перериваємо внутрішній цикл, щоб почекати і спробувати всі квері наново

                current_metrics[resource] = val

            if all_metrics_ok:
                fetch_success = True
                break  # Всі метрики зібрано успішно, виходимо з циклу ретраїв

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        # Якщо після всіх спроб дані не зібрано - скасовуємо цикл
        if not fetch_success:
            await send_system_log(
                "❌ Prometheus недоступний. Пропускаємо цей цикл, щоб не псувати історію нулями.",
                level="ERROR",
                service="collector",
            )
            return

        # 2. Якщо все успішно, синхронізуємо actual_value для минулих прогнозів
        for resource, val in current_metrics.items():
            await api_client.sync_actual_values(resource, val)

        # 3. Додаємо точку в історію
        point = {
            "cpu": current_metrics.get("cpu", 0.0),
            "ram": current_metrics.get("ram", 0.0),
            "rps": current_metrics.get("rps", 0.0),
        }
        history_buffer.append(point)
        await send_system_log(
            f"   📥 Зібрано: CPU={point['cpu']:.2f}, RAM={point['ram']:.2f}, RPS={point['rps']:.2f}",
            level="INFO",
            service="collector",
        )

        # 4. Якщо є 10 точок - робимо прогноз
        if len(history_buffer) == 10:
            payload = list(history_buffer)
            predictions = await predictor.get_prediction(payload)

            if predictions:
                await send_system_log(
                    f"   🔮 Прогноз: CPU={predictions['cpu']:.2f}, RAM={predictions['ram']:.2f}, RPS={predictions['rps']:.2f}",
                    level="INFO",
                    service="collector",
                )

                # Віддаємо прогнози в Prometheus
                PREDICTED_CPU.set(predictions["cpu"])
                PREDICTED_RAM.set(predictions["ram"])
                PREDICTED_RPS.set(predictions["rps"])

                # Записуємо в базу новий прогноз
                for resource, pred_val in predictions.items():
                    await api_client.save_new_prediction(
                        resource, current_metrics[resource], pred_val
                    )
        else:
            await send_system_log(
                f"   ⏳ Накопичення історії: {len(history_buffer)}/10. Прогноз пропускаємо.",
                level="INFO",
                service="collector",
            )

    except Exception as e:
        await send_system_log(
            f"❌ Непередбачена помилка у фоновій тасці: {e}",
            level="ERROR",
            service="collector",
        )
    finally:
        is_busy = False


async def main() -> None:
    await send_system_log(
        "🚀 Collector Service запущено у Динамічному режимі.",
        level="INFO",
        service="collector",
    )

    # Запускаємо сервер метрик Prometheus на порту 8001
    start_http_server(8001, addr="0.0.0.0")
    await send_system_log(
        "📡 Prometheus Exporter запущено на порту 8001 (/metrics)",
        level="INFO",
        service="collector",
    )
    await restore_history_buffer()
    loop = asyncio.get_event_loop()
    next_run_time = loop.time()
    interval = 15

    while True:
        try:
            # 1. Запитуємо свіжі налаштування з нашого API!
            sys_settings = await api_client.get_system_settings()
            is_active = sys_settings.get("is_collector_active", True)
        except Exception as e:

            await send_system_log(
                f"⚠️ Не вдалося отримати налаштування з API ({e}). Використовуються локальні дефолти.",
                level="WARNING",
                service="collector",
            )
            await asyncio.sleep(5)
            continue

        # 2. Перевіряємо, чи ввімкнений колектор
        if is_active:
            asyncio.create_task(process_metrics_task(sys_settings))
        else:
            await send_system_log(
                f"⏸️ Датаколектор вимкнено через Дашборд. Чекаємо {interval} сек...",
                level="INFO",
                service="collector",
            )

        # 3. Розумний сліп (враховує можливі зміни інтервалу)
        next_run_time += interval
        sleep_time = next_run_time - loop.time()

        # Якщо скрипт "забуксував" або інтервал різко зменшили
        if sleep_time <= 0:
            next_run_time = loop.time()
            sleep_time = interval

        await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    asyncio.run(main())
