from locust import HttpUser, task, between, LoadTestShape
import math
import random

class WebsiteUser(HttpUser):
    # 1. Зменшуємо час очікування! Тепер юзери "клікають" як божевільні
    wait_time = between(0.01, 0.1)

    @task(3) # Вага 3: 75% часу робимо звичайний швидкий GET
    def access_endpoint(self):
        self.client.get("/")

    @task(1) # Вага 1: 25% часу вантажимо процесор парсингом JSON
    def heavy_post_request(self):
        # Генеруємо "сміттєвий" payload на ~5 КБ, щоб змусити podinfo витрачати CPU
        payload = {
            "user_id": random.randint(1, 10000),
            "data": "A" * 5000, 
            "status": "testing"
        }
        # podinfo підтримує /echo - він парсить JSON і віддає його назад
        self.client.post("/echo", json=payload)

class DiplomaLoadShape(LoadTestShape):
    """
    Агресивний сценарій навантаження
    """
    
    stages = [
        # Збільшуємо кількість користувачів та швидкість їх появи
        {"duration": 300, "users": 50, "spawn_rate": 5},    # База
        {"duration": 900, "users": 800, "spawn_rate": 10},  # Агресивний ріст до стресу
        {"duration": 1500, "users": 800, "spawn_rate": 10}, # Утримання піку
        {"duration": 2400, "users": 300, "spawn_rate": 10}, # Спад та хвилі
        {"duration": 3000, "users": 20, "spawn_rate": 5}    # Охолодження
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                # Робимо амплітуду хвиль більшою (150 замість 50)
                if 1500 < run_time < 2400:
                    variation = 150 * math.sin(run_time / 60)
                    tick_users = stage["users"] + int(variation)
                else:
                    tick_users = stage["users"]
                
                return (tick_users, stage["spawn_rate"])

        return None
