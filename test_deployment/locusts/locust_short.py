import random

from locust import FastHttpUser, LoadTestShape, between, task


# ЗМІНА 1: Використовуємо FastHttpUser замість HttpUser!
class CPUKillerUser(FastHttpUser):
    # Затримка достатня, щоб комп'ютер дихав, але трафік був жорстким
    wait_time = between(0.1, 0.5)

    @task(3)
    def brutal_get_flood(self) -> None:
        self.client.get(f"/?cb={random.randint(1, 1000000)}", name="/ (Brutal GET)")

    @task(1)
    def heavy_json_parsing(self) -> None:
        size = random.randint(50000, 100000)
        payload = {
            "session_id": f"stress_bot_{random.randint(1, 999999)}",
            "metrics": [random.random() for _ in range(100)],
            "junk_data": "X" * size,
        }
        self.client.post("/echo", json=payload, name="/echo (Heavy POST)")


# ЗМІНА 2: Чистий 5-хвилинний тест без агресивних вбивств потоків
class CleanABTestShape(LoadTestShape):
    """
    Ідеальний 5-хвилинний сценарій для дипломного A/B тестування.
    Чітко показує різницю у швидкості реакції HPA та KEDA.
    """

    def tick(self) -> tuple[int, int] | None:
        run_time = self.get_run_time()

        if run_time >= 300:  # Зупиняємо рівно через 5 хвилин
            return None

        # ----------------------------------------------------------------
        # ФАЗА 1: "Розігрів" (0 - 1 хвилина)
        # ----------------------------------------------------------------
        if run_time < 60:
            return (50, 5)  # 50 юзерів, плавний старт

        # ----------------------------------------------------------------
        # ФАЗА 2: "Різкий удар" (1 - 4 хвилина)
        # Злітаємо до 600 юзерів.
        # (600 юзерів на FastHttpUser дадуть більше навантаження, ніж 1500 на звичайному!)
        # ----------------------------------------------------------------
        if run_time < 240:
            return (600, 50)

        # ----------------------------------------------------------------
        # ФАЗА 3: "Плавний відтік" (4 - 5 хвилина)
        # ----------------------------------------------------------------
        if run_time < 300:
            return (50, 20)
