"""
Training dataset generator: 3 повтори кожного з 5 сценаріїв у перемішаному порядку.

Порядок перемішується з фіксованим seed (ORDER_SEED=99) — відтворюваний експеримент.
Random walk отримує різний seed кожного повтору → різні траєкторії, більша різноманітність.

Загальна тривалість (~3 повтори):
    15 сценаріїв × ~610s  +  14 cooldowns × 120s  ≈  ~142 хв

Запуск:
    locust -f locustfile_training.py --host http://localhost:8080
"""

import math
import random
import time

from locust import FastHttpUser, LoadTestShape, between, task

# ── Конфігурація ───────────────────────────────────────────────────
COOLDOWN = 120  # секунд між сценаріями
REPEATS = 3  # скільки разів повторити кожен сценарій
ORDER_SEED = 99  # seed для перемішування порядку — фіксований для відтворюваності


# ── User ───────────────────────────────────────────────────────────


class CpuServiceUser(FastHttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def compute(self) -> None:
        self.client.post(
            "/compute",
            json={"data": "load_test", "rounds": 10},
            name="/compute (POST)",
        )


# ── Сценарії ───────────────────────────────────────────────────────


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
    return (max(1, int(2 + wave * 28)), 5)


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


def make_random_walk(seed: int):
    """
    Фабрика: повертає незалежну функцію random_walk з власним RNG і станом.
    Кожен повтор отримує різний seed → різні траєкторії → більша різноманітність даних.
    """
    rng = random.Random(seed)
    state = {"current": 10, "last_step": -1}

    def scenario_5_random_walk(t: float) -> tuple[int, int] | None:
        if t >= 600:
            return None
        step = int(t // 10)
        if step != state["last_step"]:
            state["last_step"] = step
            delta = rng.randint(-3, 4)
            state["current"] = max(1, min(30, state["current"] + delta))
        return (state["current"], 10)

    return scenario_5_random_walk


_burst_ranges = [(150, 180, 28), (300, 340, 30), (460, 485, 26)]


def scenario_6_quiet_burst(t: float) -> tuple[int, int] | None:
    """Baseline 1, три бурсти: 28/30/26"""
    if t >= 600:
        return None
    for start, end, peak in _burst_ranges:
        if start <= t < end:
            return (peak, 30)
    return (1, 5)


# ── Побудова розкладу ──────────────────────────────────────────────


def build_schedule(
    repeats: int, order_seed: int
) -> list[tuple[float, float, str, object]]:
    """
    Створює перемішаний список (repeats × 6 сценаріїв).
    Random walk: seed = 42 + rep*17 → rep 1: seed=42, rep 2: seed=59, rep 3: seed=76
    """
    all_scenarios = []
    for rep in range(repeats):
        rw_seed = 42 + rep * 17
        all_scenarios.extend(
            [
                (
                    f"1. Linear Ramp   [rep {rep + 1}/{repeats}]",
                    600,
                    scenario_1_linear_ramp,
                ),
                (
                    f"2. Step Function [rep {rep + 1}/{repeats}]",
                    600,
                    scenario_2_step_function,
                ),
                (
                    f"3. Sine Wave     [rep {rep + 1}/{repeats}]",
                    720,
                    scenario_3_sine_wave,
                ),
                (
                    f"4. Multi Spike   [rep {rep + 1}/{repeats}]",
                    600,
                    scenario_4_multi_spike,
                ),
                (
                    f"5. Random Walk   [rep {rep + 1}/{repeats}]",
                    600,
                    make_random_walk(rw_seed),
                ),
                (
                    f"6. Quiet + Burst [rep {rep + 1}/{repeats}]",
                    600,
                    scenario_6_quiet_burst,
                ),
            ]
        )

    random.Random(order_seed).shuffle(all_scenarios)

    schedule = []
    offset = 0.0
    for i, (name, duration, func) in enumerate(all_scenarios):
        schedule.append((offset, offset + duration, name, func))
        offset += duration
        if i < len(all_scenarios) - 1:
            offset += COOLDOWN

    total_min = offset / 60
    print(
        f"\nSchedule built: {len(all_scenarios)} scenarios, ~{total_min:.0f} min total"
    )
    print("Order:")
    for start, end, name, _ in schedule:
        print(f"  {start/60:5.1f}-{end/60:5.1f} min  {name}")

    return schedule


# ── LoadTestShape ──────────────────────────────────────────────────


class TrainingDataShape(LoadTestShape):
    def __init__(self) -> None:
        super().__init__()
        self.schedule = build_schedule(REPEATS, ORDER_SEED)
        self.total_duration = self.schedule[-1][1]
        self._last_scenario = ""

    def tick(self) -> tuple[int, int] | None:
        t = self.get_run_time()

        if t >= self.total_duration:
            return None

        for start, end, name, func in self.schedule:
            if start <= t < end:
                if name != self._last_scenario:
                    self._last_scenario = name
                    print(f"\n{'=' * 55}")
                    print(f"  SCENARIO: {name}")
                    print(f"  Time: {time.strftime('%H:%M:%S')}  (t={t:.0f}s)")
                    print(f"{'=' * 55}\n")
                local_t = t - start
                result = func(local_t)
                return result if result else (0, 1)

        # cooldown між сценаріями
        return (0, 200)
