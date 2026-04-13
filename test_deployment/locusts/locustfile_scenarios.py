"""
6 сценаріїв навантаження для cpu-service.
Запускаються послідовно з 2-хвилинним cooldown між ними.
Загальна тривалість: ~72 хвилини (6 сценаріїв × ~10-12 хв + 5 cooldown × 2 хв).

Масштаб users розраховано під bcrypt rounds=10 (~100ms/req) і CPU limit 2000m:
  - 1 core безперервно = ~10 RPS = ~10 users при wait_time(0.5, 1.5)
  - max 30 users ≈ ~1.5-1.8 cores (90% від ліміту)

Запуск:
    locust -f locustfile_scenarios.py --host http://localhost:8080
"""

import math
import random
import time

from locust import FastHttpUser, LoadTestShape, between, task

COOLDOWN = 120


class CpuServiceUser(FastHttpUser):
    wait_time = between(0.5, 1.5)  # повільніший темп — CPU не перевантажується

    @task
    def compute(self) -> None:
        self.client.post(
            "/compute",
            json={"data": "load_test", "rounds": 10},
            name="/compute (POST)",
        )


# ── Визначення сценаріїв (масштаб: max ~30 users) ──────────────────


def scenario_1_linear_ramp(t: float) -> tuple[int, int] | None:
    """0-3 хв: 1→30, 3-7 хв: plateau 30, 7-10 хв: 30→1"""
    if t < 180:
        return (max(1, int(1 + (t / 180) * 29)), 2)
    if t < 420:
        return (30, 2)
    if t < 600:
        return (max(1, int(30 - ((t - 420) / 180) * 29)), 2)
    return None


def scenario_2_step_function(t: float) -> tuple[int, int] | None:
    """Різкі стрибки: 1→25→5→30→10→1"""
    steps = [(120, 1), (240, 25), (300, 5), (420, 30), (510, 10), (600, 1)]
    for end_time, users in steps:
        if t < end_time:
            return (users, 30)
    return None


def scenario_3_sine_wave(t: float) -> tuple[int, int] | None:
    """sin² хвиля: 2-30 users, період 180с, 4 цикли = 720с"""
    if t >= 720:
        return None
    wave = math.sin(t * 2 * math.pi / 180) ** 2
    users = 2 + wave * 28
    return (max(1, int(users)), 5)


_spike_ranges = [
    (60, 80, 28),
    (150, 170, 30),
    (230, 250, 25),
    (330, 355, 30),
    (410, 425, 22),
    (490, 515, 28),
    (560, 580, 26),
]


def scenario_4_multi_spike(t: float) -> tuple[int, int] | None:
    """Базове 2, спайки 22-30 users по 20с"""
    if t >= 600:
        return None
    for start, end, peak in _spike_ranges:
        if start <= t < end:
            return (peak, 30)
    return (2, 5)


_rng = random.Random(42)
_rw_state = {"current": 10, "last_step": -1}


def scenario_5_random_walk(t: float) -> tuple[int, int] | None:
    """Випадкова зміна ±3 кожні 10с, seed=42, діапазон 1-30"""
    if t >= 600:
        return None
    step = int(t // 10)
    if step != _rw_state["last_step"]:
        _rw_state["last_step"] = step
        delta = _rng.randint(-3, 4)
        _rw_state["current"] = max(1, min(30, _rw_state["current"] + delta))
    return (_rw_state["current"], 10)


_burst_ranges = [(150, 180, 28), (300, 340, 30), (460, 485, 26)]


def scenario_6_quiet_burst(t: float) -> tuple[int, int] | None:
    """Baseline 1, три бурсти: 28/30/26"""
    if t >= 600:
        return None
    for start, end, peak in _burst_ranges:
        if start <= t < end:
            return (peak, 30)
    return (1, 5)


# ── Список сценаріїв ───────────────────────────────────────────────

SCENARIOS = [
    ("1. Linear Ramp", 600, scenario_1_linear_ramp),
    ("2. Step Function", 600, scenario_2_step_function),
    ("3. Sine Wave", 720, scenario_3_sine_wave),
    ("4. Multi Spike", 600, scenario_4_multi_spike),
    ("5. Random Walk", 600, scenario_5_random_walk),
    ("6. Quiet + Burst", 600, scenario_6_quiet_burst),
]


# ── LoadTestShape ──────────────────────────────────────────────────


class AllScenariosShape(LoadTestShape):
    def __init__(self) -> None:
        super().__init__()
        self.schedule: list[tuple[float, float, str, object]] = []
        offset = 0.0
        for i, (name, duration, func) in enumerate(SCENARIOS):
            self.schedule.append((offset, offset + duration, name, func))
            offset += duration
            if i < len(SCENARIOS) - 1:
                offset += COOLDOWN
        self.total_duration = offset
        self._last_scenario = ""

    def tick(self) -> tuple[int, int] | None:
        t = self.get_run_time()

        if t >= self.total_duration:
            return None

        for start, end, name, func in self.schedule:
            if start <= t < end:
                if name != self._last_scenario:
                    self._last_scenario = name
                    print(f"\n{'='*50}")
                    print(f"  SCENARIO: {name}")
                    print(f"  Time: {time.strftime('%H:%M:%S')}")
                    print(f"{'='*50}\n")
                local_t = t - start
                result = func(local_t)
                return result if result else (0, 1)

        if self._find_cooldown(t):
            return (0, 200)

        return None

    def _find_cooldown(self, t: float) -> bool:
        for i in range(len(self.schedule) - 1):
            _, end_current, _, _ = self.schedule[i]
            start_next, _, _, _ = self.schedule[i + 1]
            if end_current <= t < start_next:
                return True
        return False
