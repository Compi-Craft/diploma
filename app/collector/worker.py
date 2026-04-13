import asyncio
import datetime
from collections import deque
from typing import Deque, Dict, Tuple

from config import MODEL_INPUT_STEPS
from prometheus_client import Gauge, start_http_server
from shared.logger import send_system_log
from shared.schemas import SettingsRead

from .services import api_client, prometheus

PREDICTED_CPU = Gauge(
    "gru_predicted_cpu_util",
    "Predicted CPU utilization [0,1] for the next window (GRU)",
)

# Below this actual CPU utilization, the input is outside the training distribution
# (training data min was ~0.028 under Locust load). A linear blend smoothly shifts
# the exposed prediction toward actual_cpu, preventing idle over-provisioning.
OOD_BLEND_THRESHOLD = 0.10

is_busy = False
history_buffer: Deque[Dict[str, float]] = deque(maxlen=MODEL_INPUT_STEPS + 1)
# Черга: (row_id, target_ts) — для точного sync actual_cpu по id
pending_sync: Deque[Tuple[int, datetime.datetime]] = deque()


async def restore_pending_sync() -> None:
    """При рестарті відновлює чергу sync для рядків, які ще не отримали actual_cpu.
    Оскільки sync використовує raw_cpu_readings, значення будуть точними навіть
    для рядків де target_ts вже давно минув."""
    try:
        orphans = await api_client.get_unsynced_rows()
        for row in orphans:
            pending_sync.append((row.id, row.target_ts))
        if orphans:
            await send_system_log(
                f"🔄 Відновлено {len(orphans)} рядків у pending_sync після рестарту.",
                level="INFO",
                service="collector",
            )
    except Exception as e:
        await send_system_log(
            f"⚠️ Помилка відновлення pending_sync: {e}",
            level="ERROR",
            service="collector",
        )


async def restore_history_buffer() -> None:
    """Відновлює останні MODEL_INPUT_STEPS + 1 точок з бази даних для швидкого старту GRU."""
    await send_system_log(
        "🔄 Спроба відновити історію з БД для швидкого старту...",
        level="INFO",
        service="collector",
    )
    try:
        recent_data = await api_client.get_recent_history(limit=MODEL_INPUT_STEPS + 1)

        if not recent_data:
            await send_system_log(
                "ℹ️ База порожня. Буфер почне заповнюватися з нуля.",
                level="INFO",
                service="collector",
            )
            return

        # API повертає DESC, нам потрібен хронологічний порядок
        recent_data.reverse()

        for entry in recent_data:
            point = {
                "cpu": entry.input_cpu if entry.input_cpu is not None else 0.0,
                "rps": entry.input_rps if entry.input_rps is not None else 0.0,
            }
            history_buffer.append(point)

        await send_system_log(
            f"✅ Буфер відновлено! Завантажено {len(recent_data)}/{MODEL_INPUT_STEPS + 1} точок з БД.",
            level="INFO",
            service="collector",
        )

    except Exception as e:
        await send_system_log(
            f"⚠️ Помилка відновлення буфера: {e}", "ERROR", "collector"
        )


async def process_metrics_task(sys_settings: SettingsRead) -> None:
    global is_busy
    if is_busy:
        return

    is_busy = True
    await send_system_log("🕒 Початок збору метрик", level="INFO", service="collector")

    try:
        prom_url: str = sys_settings.prometheus_url or ""
        queries = {
            "cpu": sys_settings.cpu_query,
            "ram": sys_settings.ram_query,
            "rps": sys_settings.rps_query,
        }

        current_metrics = {}
        max_retries = 3
        retry_delay = 0.5
        fetch_success = False

        for attempt in range(max_retries):
            all_metrics_ok = True

            for resource, query in queries.items():
                if not query:
                    continue

                try:
                    val = await prometheus.fetch_metric(query, prom_url=prom_url)
                except Exception as e:
                    val = None

                if val is None:
                    all_metrics_ok = False
                    break

                current_metrics[resource] = val
                await asyncio.sleep(0.1)

            if all_metrics_ok:
                fetch_success = True
                break

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        if not fetch_success:
            if history_buffer:
                history_buffer.append(history_buffer[-1])
                await send_system_log(
                    "⚠️ Prometheus недоступний. Forward-fill: використано попередню точку для збереження 5s інтервалу.",
                    level="WARNING",
                    service="collector",
                )
            else:
                await send_system_log(
                    "❌ Prometheus недоступний і буфер порожній. Пропускаємо цей цикл.",
                    level="ERROR",
                    service="collector",
                )
            return

        # Зберігаємо сире вимірювання CPU з точною міткою часу
        now = datetime.datetime.now(datetime.timezone.utc)
        await api_client.save_raw_reading(ts=now, cpu_value=current_metrics["cpu"])

        # Синхронізуємо actual_cpu для рядків з черги, час яких настав
        while pending_sync and pending_sync[0][1] <= now:
            row_id, _ = pending_sync.popleft()
            await api_client.sync_by_id(row_id)

        # Додаємо точку в історію
        point = {
            "cpu": current_metrics.get("cpu", 0.0),
            "rps": current_metrics.get("rps", 0.0),
        }
        history_buffer.append(point)
        await send_system_log(
            f"   📥 Зібрано: CPU_util={point['cpu']:.4f} ({point['cpu']*100:.1f}%), RPS={point['rps']:.2f}",
            level="INFO",
            service="collector",
        )

        # Прогноз ТІЛЬКИ коли зібрано MODEL_INPUT_STEPS + 1 точок
        if len(history_buffer) == MODEL_INPUT_STEPS + 1:
            payload = list(history_buffer)
            predicted_cpu = await api_client.get_prediction(payload)

            if predicted_cpu is not None:
                cpu_limit = sys_settings.prediction_cpu_limit
                capped = min(predicted_cpu, cpu_limit)

                # OOD guard: blend toward actual when CPU is below training distribution.
                # Training data never contained cpu_util < 0.028; below OOD_BLEND_THRESHOLD
                # the model is unreliable and saturates artificially high.
                actual_cpu = current_metrics["cpu"]
                alpha = min(1.0, actual_cpu / OOD_BLEND_THRESHOLD)
                exposed = alpha * capped + (1.0 - alpha) * actual_cpu

                await send_system_log(
                    f"   🔮 Прогноз CPU util: raw={predicted_cpu:.4f} ({predicted_cpu*100:.1f}%), "
                    f"ood_alpha={alpha:.3f}, exposed={exposed:.4f} (limit={cpu_limit})",
                    level="INFO",
                    service="collector",
                )

                PREDICTED_CPU.set(exposed)

                saved = await api_client.save_new_prediction(
                    input_cpu=current_metrics["cpu"],
                    input_ram=current_metrics["ram"],
                    input_rps=current_metrics["rps"],
                    predicted_cpu=predicted_cpu,
                )
                if saved:
                    pending_sync.append((saved.id, saved.target_ts))
        else:
            await send_system_log(
                f"   ⏳ Накопичення історії: {len(history_buffer)}/{MODEL_INPUT_STEPS + 1}. Прогноз пропускаємо.",
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

    start_http_server(8001, addr="0.0.0.0")
    await send_system_log(
        "📡 Prometheus Exporter запущено на порту 8001 (/metrics)",
        level="INFO",
        service="collector",
    )
    await restore_history_buffer()
    await restore_pending_sync()
    loop = asyncio.get_event_loop()
    next_run_time = loop.time()
    interval = 5

    while True:
        sys_settings = await api_client.get_system_settings()
        is_active = sys_settings.is_collector_active

        if is_active:
            asyncio.create_task(process_metrics_task(sys_settings))
        else:
            await send_system_log(
                f"⏸️ Датаколектор вимкнено через Дашборд. Чекаємо {interval} сек...",
                level="INFO",
                service="collector",
            )

        next_run_time += interval
        sleep_time = next_run_time - loop.time()

        if sleep_time <= 0:
            next_run_time = loop.time()
            sleep_time = interval

        await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    asyncio.run(main())
