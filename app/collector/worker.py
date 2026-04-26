import asyncio
import datetime
from collections import deque
from typing import Deque, Dict, Tuple

from prometheus_client import Gauge, start_http_server
from shared.logger import send_system_log
from shared.schemas import SettingsRead

from .services import api_client, prometheus

PREDICTED_CPU = Gauge(
    "gru_predicted_cpu_util",
    "Predicted CPU utilization [0,1] for the next window (GRU)",
)

COLLECTION_INTERVAL_SEC = 5  # seconds between Prometheus scrapes

# History buffer holds more points than any expected window size.
# The collector slices the last window_size+1 points before each prediction call.
MAX_BUFFER_SIZE = 51

is_busy = False
history_buffer: Deque[Dict[str, float]] = deque(maxlen=MAX_BUFFER_SIZE)
# Queue: (row_id, target_ts) — for precise actual_cpu sync by id
pending_sync: Deque[Tuple[int, datetime.datetime]] = deque()


async def restore_pending_sync() -> None:
    """On restart, restore sync queue for rows that have not yet received actual_cpu."""
    try:
        orphans = await api_client.get_unsynced_rows()
        for row in orphans:
            pending_sync.append((row.id, row.target_ts))
        if orphans:
            await send_system_log(
                f"Restored {len(orphans)} rows in pending_sync after restart.",
                level="INFO",
                service="collector",
            )
    except Exception as e:
        await send_system_log(
            f"Failed to restore pending_sync: {e}",
            level="ERROR",
            service="collector",
        )


async def restore_history_buffer() -> None:
    """Load recent DB records into the buffer so GRU can start predicting immediately."""
    await send_system_log(
        "Attempting to restore history buffer from DB...",
        level="INFO",
        service="collector",
    )
    try:
        recent_data = await api_client.get_recent_history(limit=MAX_BUFFER_SIZE)

        if not recent_data:
            await send_system_log(
                "DB is empty. Buffer will fill from scratch.",
                level="INFO",
                service="collector",
            )
            return

        # API returns DESC order; restore in chronological order
        recent_data.reverse()

        for entry in recent_data:
            point = {
                "cpu": entry.input_cpu if entry.input_cpu is not None else 0.0,
                "rps": entry.input_rps if entry.input_rps is not None else 0.0,
            }
            history_buffer.append(point)

        await send_system_log(
            f"Buffer restored: {len(recent_data)}/{MAX_BUFFER_SIZE} points loaded.",
            level="INFO",
            service="collector",
        )

    except Exception as e:
        await send_system_log(f"Failed to restore buffer: {e}", "ERROR", "collector")


async def process_metrics_task(sys_settings: SettingsRead) -> None:
    global is_busy
    if is_busy:
        return

    is_busy = True
    await send_system_log(
        "Metric collection cycle started", level="INFO", service="collector"
    )

    try:
        prom_url: str = sys_settings.prometheus_url or ""
        queries = {
            "cpu": sys_settings.cpu_query,
            "ram": sys_settings.ram_query,
            "rps": sys_settings.rps_query,
        }

        current_metrics: Dict[str, float] = {}
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
                except Exception:
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
                    "Prometheus unavailable. Forward-fill: reusing last point to preserve 5s interval.",
                    level="WARNING",
                    service="collector",
                )
            else:
                await send_system_log(
                    "Prometheus unavailable and buffer is empty. Skipping this cycle.",
                    level="ERROR",
                    service="collector",
                )
            return

        now = datetime.datetime.now(datetime.timezone.utc)
        await api_client.save_raw_reading(ts=now, cpu_value=current_metrics["cpu"])

        while pending_sync and pending_sync[0][1] <= now:
            row_id, _ = pending_sync.popleft()
            await api_client.sync_by_id(row_id)

        point = {
            "cpu": current_metrics.get("cpu", 0.0),
            "rps": current_metrics.get("rps", 0.0),
        }
        history_buffer.append(point)
        await send_system_log(
            f"   Collected: CPU_util={point['cpu']:.4f} ({point['cpu']*100:.1f}%), RPS={point['rps']:.2f}",
            level="INFO",
            service="collector",
        )

        window_size, forecast_horizon = await api_client.get_predictor_model_config()
        horizon_seconds = forecast_horizon * COLLECTION_INTERVAL_SEC

        if len(history_buffer) >= window_size + 1:
            payload = list(history_buffer)[-(window_size + 1) :]
            predicted_cpu = await api_client.get_prediction(payload)

            if predicted_cpu is not None:
                ood_threshold = sys_settings.ood_blend_threshold

                # OOD guard: linearly blend toward actual_cpu when below training distribution.
                # Training data min was ~0.028 under Locust load; below the threshold the model
                # output saturates artificially high at idle.
                actual_cpu = current_metrics["cpu"]
                alpha = (
                    min(1.0, actual_cpu / ood_threshold) if ood_threshold > 0 else 1.0
                )
                exposed = alpha * predicted_cpu + (1.0 - alpha) * actual_cpu

                await send_system_log(
                    f"   Prediction: raw={predicted_cpu:.4f} ({predicted_cpu*100:.1f}%), "
                    f"ood_alpha={alpha:.3f}, exposed={exposed:.4f}, "
                    f"horizon={horizon_seconds}s (fh={forecast_horizon}×{COLLECTION_INTERVAL_SEC}s)",
                    level="INFO",
                    service="collector",
                )

                PREDICTED_CPU.set(exposed)

                saved = await api_client.save_new_prediction(
                    input_cpu=current_metrics["cpu"],
                    input_ram=current_metrics["ram"],
                    input_rps=current_metrics["rps"],
                    predicted_cpu=predicted_cpu,
                    horizon_seconds=horizon_seconds,
                )
                if saved:
                    pending_sync.append((saved.id, saved.target_ts))
        else:
            await send_system_log(
                f"   Accumulating history: {len(history_buffer)}/{window_size + 1}. Skipping prediction.",
                level="INFO",
                service="collector",
            )

    except Exception as e:
        await send_system_log(
            f"Unexpected error in metric collection task: {e}",
            level="ERROR",
            service="collector",
        )
    finally:
        is_busy = False


async def main() -> None:
    await send_system_log(
        "Collector service started.",
        level="INFO",
        service="collector",
    )

    start_http_server(8001, addr="0.0.0.0")
    await send_system_log(
        "Prometheus exporter started on port 8001 (/metrics)",
        level="INFO",
        service="collector",
    )
    await restore_history_buffer()
    await restore_pending_sync()
    loop = asyncio.get_event_loop()
    next_run_time = loop.time()

    while True:
        sys_settings = await api_client.get_system_settings()
        is_active = sys_settings.is_collector_active

        if is_active:
            asyncio.create_task(process_metrics_task(sys_settings))
        else:
            await send_system_log(
                f"Collector disabled via Dashboard. Waiting {COLLECTION_INTERVAL_SEC}s...",
                level="INFO",
                service="collector",
            )

        next_run_time += COLLECTION_INTERVAL_SEC
        sleep_time = next_run_time - loop.time()

        if sleep_time <= 0:
            next_run_time = loop.time()
            sleep_time = COLLECTION_INTERVAL_SEC

        await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    asyncio.run(main())
