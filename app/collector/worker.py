import asyncio
import datetime
from collections import deque
from prometheus_client import Gauge, start_http_server
from .services import prometheus, predictor, api_client

# --- Оголошуємо наші метрики ---
PREDICTED_CPU = Gauge('lstm_predicted_cpu_cores', 'Predicted CPU usage in cores for the next window')
PREDICTED_RAM = Gauge('lstm_predicted_ram_mb', 'Predicted RAM usage in MB')
PREDICTED_RPS = Gauge('lstm_predicted_rps', 'Predicted Requests Per Second')
# -------------------------------------

is_busy = False
history_buffer = deque(maxlen=10)

async def process_metrics_task(sys_settings: dict):
    global is_busy
    if is_busy:
        return

    is_busy = True
    print(f"\n🕒 Початок збору: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    try:
        current_metrics = {}
        
        prom_url = sys_settings.get("prometheus_url")
        queries = {
            "cpu": sys_settings.get("cpu_query"),
            "ram": sys_settings.get("ram_query"),
            "rps": sys_settings.get("rps_query")
        }
        
        # 1. Збираємо реальні дані
        for resource, query in queries.items():
            if not query:
                continue
            # ПЕРЕДАЄМО prom_url у fetch_metric (тобі треба буде трохи оновити цей метод у prometheus.py)
            val = await prometheus.fetch_metric(query, prom_url=prom_url)
            current_metrics[resource] = val if val is not None else 0.0
            
            # Записуємо actual_value для минулих прогнозів
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
                
                # Віддаємо прогнози в Prometheus
                PREDICTED_CPU.set(predictions['cpu'])
                PREDICTED_RAM.set(predictions['ram'])
                PREDICTED_RPS.set(predictions['rps'])
                
                # Записуємо в базу новий прогноз
                for resource, pred_val in predictions.items():
                    await api_client.save_new_prediction(resource, current_metrics[resource], pred_val)
        else:
            print(f"   ⏳ Накопичення історії: {len(history_buffer)}/10. Прогноз пропускаємо.")

    except Exception as e:
        print(f"❌ Помилка у фоновій тасці: {e}")
    finally:
        is_busy = False


async def main():
    print(f"🚀 Collector Service запущено у Динамічному режимі.")
    
    # Запускаємо сервер метрик Prometheus на порту 8001
    start_http_server(8001, addr="0.0.0.0")
    print("📡 Prometheus Exporter запущено на порту 8001 (/metrics)")
    
    loop = asyncio.get_event_loop()
    next_run_time = loop.time()

    while True:
        try:
            # 1. Запитуємо свіжі налаштування з нашого API!
            sys_settings = await api_client.get_system_settings()
            interval = sys_settings.get("collection_interval_sec", 15)
            is_active = sys_settings.get("is_collector_active", True)
        except Exception as e:
            print(f"⚠️ Не вдалося отримати налаштування з API ({e}). Використовуються локальні дефолти.")
            await asyncio.sleep(5)
            continue

        # 2. Перевіряємо, чи ввімкнений колектор
        if is_active:
            asyncio.create_task(process_metrics_task(sys_settings))
        else:
            print(f"⏸️ Датаколектор вимкнено через Дашборд. Чекаємо {interval} сек...")

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