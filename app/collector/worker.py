import asyncio
import datetime
from .config import settings
from .services import prometheus, predictor, api_client

is_busy = False

async def process_metrics_task():
    global is_busy
    if is_busy:
        print("⏩ Пропуск циклу: попередня обробка ще триває...")
        return

    is_busy = True
    print(f"🕒 Початок збору: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    try:
        for resource, query in settings.MONITORING_QUERIES.items():
            # 1. Збір даних з Prometheus
            current_val = await prometheus.fetch_metric(query)

            if current_val is not None:
                # 2. Оновлюємо реальні значення через HTTP-запит до API
                await api_client.sync_actual_values(resource, current_val)
                
                # 3. Отримуємо прогноз від LSTM (теж HTTP)
                prediction = await predictor.get_prediction(resource, current_val)
                
                # 4. Записуємо нову пару через HTTP-запит до API
                await api_client.save_new_prediction(resource, current_val, prediction)
                
                print(f"   📊 {resource.upper()}: Current={current_val:.2f}, Predicted={prediction:.2f}")
    except Exception as e:
        print(f"❌ Помилка у фоновій тасці: {e}")
    finally:
        is_busy = False # Звільняємо місце для наступного циклу

async def main():
    print(f"🚀 Collector Service (Worker) запущено у режимі API-Client.")
    print(f"⏱️ Інтервал: {settings.COLLECTION_INTERVAL} сек.")
    
    loop = asyncio.get_event_loop()
    next_run_time = loop.time()

    while True:
        # Запускаємо роботу у фоні
        asyncio.create_task(process_metrics_task())
        
        # Вираховуємо час до наступного "кратного" запуску (00, 15, 30, 45 сек)
        next_run_time += settings.COLLECTION_INTERVAL
        sleep_time = next_run_time - loop.time()
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            # Якщо робота зайняла більше 15 секунд, не спимо зовсім
            print("⚠️ Робота тривала довше інтервалу! Наздоганяємо графік...")
            next_run_time = loop.time()

if __name__ == "__main__":
    asyncio.run(main())