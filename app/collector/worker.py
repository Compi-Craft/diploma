import asyncio
import datetime
from collections import deque
from prometheus_client import Gauge, start_http_server  # <--- НОВЕ: Імпортуємо інструменти Prometheus

from .config import settings
from .services import prometheus, predictor, api_client

# --- НОВЕ: Оголошуємо наші метрики ---
# Gauge (датчик) - це тип метрики, яка може як зростати, так і падати (ідеально для CPU/RAM)
PREDICTED_CPU = Gauge('lstm_predicted_cpu_cores', 'Predicted CPU usage in cores for the next window')
PREDICTED_RAM = Gauge('lstm_predicted_ram_mb', 'Predicted RAM usage in MB')
PREDICTED_RPS = Gauge('lstm_predicted_rps', 'Predicted Requests Per Second')
# -------------------------------------

is_busy = False
history_buffer = deque(maxlen=10)

async def process_metrics_task():
    global is_busy
    if is_busy:
        return

    is_busy = True
    print(f"\n🕒 Початок збору: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    try:
        current_metrics = {}
        
        # 1. Збираємо реальні дані
        for resource, query in settings.MONITORING_QUERIES.items():
            val = await prometheus.fetch_metric(query)
            current_metrics[resource] = val if val is not None else 0.0
            await api_client.sync_actual_values(resource, current_metrics[resource])

        point = {
            "cpu": current_metrics.get("cpu", 0.0),
            "ram": current_metrics.get("ram", 0.0),
            "rps": current_metrics.get("rps", 0.0)
        }
        history_buffer.append(point)
        print(f"   📥 Зібрано: CPU={point['cpu']:.2f}, RAM={point['ram']:.2f}, RPS={point['rps']:.2f}")

        # 2. Якщо є 10 точок - робимо прогноз
        if len(history_buffer) == 10:
            payload = list(history_buffer)
            predictions = await predictor.get_prediction(payload)
            
            if predictions:
                print(f"   🔮 Прогноз: CPU={predictions['cpu']:.2f}, RAM={predictions['ram']:.2f}, RPS={predictions['rps']:.2f}")
                
                # --- НОВЕ: Віддаємо прогнози в Prometheus ---
                PREDICTED_CPU.set(predictions['cpu'])
                PREDICTED_RAM.set(predictions['ram'])
                PREDICTED_RPS.set(predictions['rps'])
                # --------------------------------------------
                
                # Записуємо в базу
                for resource, pred_val in predictions.items():
                    await api_client.save_new_prediction(resource, current_metrics[resource], pred_val)
        else:
            print(f"   ⏳ Накопичення історії: {len(history_buffer)}/10. Прогноз пропускаємо.")

    except Exception as e:
        print(f"❌ Помилка у фоновій тасці: {e}")
    finally:
        is_busy = False

async def main():
    print(f"🚀 Collector Service запущено у режимі Batch-Prediction.")
    print(f"⏱️ Інтервал: {settings.COLLECTION_INTERVAL} сек.")
    
    # --- НОВЕ: Запускаємо сервер метрик Prometheus на порту 8001 ---
    start_http_server(8001, addr="0.0.0.0")
    print("📡 Prometheus Exporter запущено на порту 8001 (/metrics)")
    # ---------------------------------------------------------------
    
    loop = asyncio.get_event_loop()
    next_run_time = loop.time()

    while True:
        asyncio.create_task(process_metrics_task())
        
        next_run_time += settings.COLLECTION_INTERVAL
        sleep_time = next_run_time - loop.time()
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            next_run_time = loop.time()

if __name__ == "__main__":
    asyncio.run(main())