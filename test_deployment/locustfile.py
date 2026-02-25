from locust import HttpUser, task, between, LoadTestShape
import math

class WebsiteUser(HttpUser):
    # Кожен користувач чекає від 0.5 до 1.5 секунди між запитами
    wait_time = between(0.5, 1.5)

    @task
    def access_endpoint(self):
        # Ми стукаємо в основний ендпоінт. 
        # podinfo автоматично генерує навантаження при обробці запитів.
        self.client.get("/")

class DiplomaLoadShape(LoadTestShape):
    """
    Сценарій навантаження для збору датасету:
    0-5 хв: Базове навантаження (20 юзерів)
    5-15 хв: Плавний ріст до піку (250 юзерів) - тут маємо вийти на 80% CPU
    15-25 хв: "Плато" високого навантаження
    25-40 хв: Хвилі (імітація нестабільного трафіку)
    40-50 хв: Спад до мінімуму
    """
    
    stages = [
        {"duration": 300, "users": 20, "spawn_rate": 5},   # База
        {"duration": 900, "users": 250, "spawn_rate": 2},  # Ріст до стресу
        {"duration": 1500, "users": 250, "spawn_rate": 5}, # Утримання піку
        {"duration": 2400, "users": 100, "spawn_rate": 5}, # Спад та хвилі
        {"duration": 3000, "users": 10, "spawn_rate": 5}   # Охолодження
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                # Якщо ми в етапі хвиль (25-40 хв), додаємо синусоїду
                if 1500 < run_time < 2400:
                    variation = 50 * math.sin(run_time / 60)
                    tick_users = stage["users"] + int(variation)
                else:
                    tick_users = stage["users"]
                
                return (tick_users, stage["spawn_rate"])

        return None