import math
import random

from locust import HttpUser, LoadTestShape, between, task


class ChaosUser(HttpUser):
    # Різна інтенсивність "кліків" для різних користувачів
    wait_time = between(0.05, 0.5)

    @task(10)
    def standard_request(self) -> None:
        """Звичайний трафік на головну"""
        self.client.get("/")

    @task(3)
    def memory_heavy_task(self) -> None:
        """Імітація важкої обробки даних (високий RAM та CPU)"""
        # Генеруємо випадковий розмір "пакета" від 1 до 50 КБ
        payload_size = random.randint(1000, 50000)
        payload = {"data": "X" * payload_size, "type": "synthetic_load"}
        self.client.post("/echo", json=payload)

    @task(1)
    def error_prone_task(self) -> None:
        """Імітація помилок або довгих відповідей (таймаути)"""
        # podinfo має ендпоінт /delay/{seconds}
        self.client.get("/delay/2", name="/delay")


class LongTermChaosShape(LoadTestShape):
    """
    Сценарій, що імітує добу життя сервісу за 8 годин (28800 сек):
    1. Morning Rush (0-1 год): Плавний ріст.
    2. Stability (1-3 год): Рівне навантаження з шумом.
    3. Flash Crowd (3-3.5 год): Різкий пік (маркетингова акція).
    4. Fractal Noise (3.5-6 год): Хаотичні хвилі.
    5. Night Mode (6-8 год): Майже нуль запитів.
    """

    def tick(self) -> tuple[int, int] | None:
        run_time = self.get_run_time()

        # 1. Ранок: Плавний підйом (20 -> 400 юзерів)
        if run_time < 3600:
            users = 20 + (run_time / 3600) * 380
            return (int(users), 2)

        # 2. Робочий день: Стабільність (400) + синусоїдальні хвилі (+/- 50)
        elif run_time < 10800:
            variation = 50 * math.sin(run_time / 600)
            users = 400 + variation
            return (int(users), 1)

        # 3. Flash Crowd: Сплеск до 1500 за 5 хвилин і спад
        elif run_time < 12600:
            if run_time < 11700:  # Ріст
                users = 400 + ((run_time - 10800) / 900) * 1100
            else:  # Спад
                users = 1500 - ((run_time - 11700) / 900) * 1100
            return (int(users), 15)

        # 4. Фрактальний шум: Хаотичні стрибки (від 100 до 800)
        elif run_time < 21600:
            # Комбінація двох синусоїд різної частоти для "непередбачуваності"
            wave = 200 * math.sin(run_time / 300) + 100 * math.cos(run_time / 50)
            users = 450 + wave
            return (max(10, int(users)), 5)

        # 5. Ніч: Охолодження (10-20 юзерів)
        elif run_time < 28800:
            return (15, 1)

        return None
