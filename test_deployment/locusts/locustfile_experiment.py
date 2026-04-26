"""
A/B experiment locustfile — identical load shape to locustfile_scenarios.py.
Додатково: зберігає точні Unix timestamps кожного сценарію у experiment_timestamps.json
для подальшого вирізання з Prometheus і підрахунку зайвих подів.

Запуск Round A (Vanilla HPA):
    EXPERIMENT_ROUND=round_a locust -f locustfile_experiment.py \
           --host http://localhost:8080 --web-port 8089 \
           --csv results/round_a

Запуск Round B (Predictive KEDA):
    EXPERIMENT_ROUND=round_b locust -f locustfile_experiment.py \
           --host http://localhost:8080 --web-port 8089 \
           --csv results/round_b

Після завершення обох раундів: experiment_timestamps.json містить межі кожного
сценарію для обох раундів — використовуй для Prometheus range queries.
"""

import json
import math
import os
import random
import threading
import time
from pathlib import Path

from locust import FastHttpUser, LoadTestShape, between, events, task

COOLDOWN = 300
SCRIPT_DIR = Path(__file__).parent
TIMESTAMPS_FILE = SCRIPT_DIR / "experiment_timestamps.json"
RESULTS_DIR = SCRIPT_DIR / "results"

# Створюємо results/ одразу при імпорті — до того як locust спробує писати CSV
RESULTS_DIR.mkdir(exist_ok=True)

# Перевірка EXPERIMENT_ROUND при старті
_round_label = os.environ.get("EXPERIMENT_ROUND", "")
if not _round_label:
    print("\n" + "!" * 60)
    print("  ПОМИЛКА: змінна EXPERIMENT_ROUND не встановлена!")
    print("  Запускай так:")
    print("    EXPERIMENT_ROUND=round_a locust -f locustfile_experiment.py ...")
    print("!" * 60 + "\n")
    raise SystemExit(1)


# ── HTTP User ─────────────────────────────────────────────────────────


class CpuServiceUser(FastHttpUser):
    wait_time = between(0.2, 0.6)

    @task
    def compute(self) -> None:
        self.client.post(
            "/compute",
            json={"data": "load_test", "rounds": 10},
            name="/compute (POST)",
        )


# ── Сценарії (ідентичні locustfile_scenarios.py) ──────────────────────


def scenario_1_linear_ramp(t: float) -> tuple[int, int] | None:
    if t < 180:
        return (max(1, int(1 + (t / 180) * 119)), 3)
    if t < 420:
        return (120, 3)
    if t < 600:
        return (max(1, int(120 - ((t - 420) / 180) * 119)), 3)
    return None


def scenario_2_step_function(t: float) -> tuple[int, int] | None:
    steps = [(120, 3), (240, 100), (300, 20), (420, 120), (510, 40), (600, 3)]
    for end_time, users in steps:
        if t < end_time:
            return (users, 120)
    return None


def scenario_3_sine_wave(t: float) -> tuple[int, int] | None:
    if t >= 720:
        return None
    wave = math.sin(t * 2 * math.pi / 180) ** 2
    return (max(1, int(2 + wave * 118)), 15)


_spike_ranges = [
    (60, 80,  114),
    (150, 170, 120),
    (230, 250, 102),
    (330, 355, 120),
    (410, 425,  90),
    (490, 515, 114),
    (560, 580, 105),
]


def scenario_4_multi_spike(t: float) -> tuple[int, int] | None:
    if t >= 600:
        return None
    for start, end, peak in _spike_ranges:
        if start <= t < end:
            return (peak, 120)
    return (9, 15)


_rng = random.Random(42)
_rw_state = {"current": 42, "last_step": -1}


def scenario_5_random_walk(t: float) -> tuple[int, int] | None:
    if t >= 600:
        return None
    step = int(t // 10)
    if step != _rw_state["last_step"]:
        _rw_state["last_step"] = step
        delta = _rng.randint(-9, 12)
        _rw_state["current"] = max(1, min(120, _rw_state["current"] + delta))
    return (_rw_state["current"], 42)


_burst_ranges = [(150, 180, 114), (300, 340, 120), (460, 485, 105)]


def scenario_6_quiet_burst(t: float) -> tuple[int, int] | None:
    if t >= 600:
        return None
    for start, end, peak in _burst_ranges:
        if start <= t < end:
            return (peak, 120)
    return (1, 7)


SCENARIOS = [
    ("1. Linear Ramp", 600, scenario_1_linear_ramp),
    ("2. Step Function", 600, scenario_2_step_function),
    ("3. Sine Wave", 720, scenario_3_sine_wave),
    ("4. Multi Spike", 600, scenario_4_multi_spike),
    ("5. Random Walk", 600, scenario_5_random_walk),
    ("6. Quiet + Burst", 600, scenario_6_quiet_burst),
]


# ── LoadTestShape з логуванням timestamps ────────────────────────────


class ExperimentShape(LoadTestShape):
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
        self._wall_start: float | None = None
        self._round_log: dict = {}

        self._round = _round_label
        self._timestamps_saved = False
        self._quitting = False

        # Зберігаємо timestamps при Ctrl+C / передчасному завершенні
        events.quitting.add_listener(self._on_quit)

    def tick(self) -> tuple[int, int] | None:
        t = self.get_run_time()

        if self._wall_start is None:
            self._wall_start = time.time() - t

        if t >= self.total_duration:
            self._save_timestamps()
            if not self._quitting and self.runner is not None:
                self._quitting = True
                threading.Thread(
                    target=self.runner.quit, daemon=True
                ).start()
            return None

        for start, end, name, func in self.schedule:
            if start <= t < end:
                if name != self._last_scenario:
                    wall_ts = self._wall_start + start
                    self._last_scenario = name
                    self._round_log[name] = {
                        "start_unix": wall_ts,
                        "end_unix": self._wall_start + end,
                        "start_str": time.strftime("%H:%M:%S", time.localtime(wall_ts)),
                    }
                    print(f"\n{'='*50}")
                    print(f"  SCENARIO: {name}  [{time.strftime('%H:%M:%S')}]")
                    print(f"{'='*50}\n")
                    # Зберігаємо після кожного сценарію — якщо впаде в середині, дані не загубляться
                    self._save_timestamps()

                local_t = t - start
                result = func(local_t)
                return result if result else (0, 1)

        return (0, 200)

    def _on_quit(self, **kwargs) -> None:
        if not self._timestamps_saved:
            self._save_timestamps()

    def _save_timestamps(self) -> None:
        if not self._round_log:
            return
        try:
            data: dict = {}
            if TIMESTAMPS_FILE.exists():
                try:
                    data = json.loads(TIMESTAMPS_FILE.read_text())
                except Exception:
                    backup = TIMESTAMPS_FILE.with_suffix(".bak.json")
                    TIMESTAMPS_FILE.rename(backup)
                    print(f"⚠️  Пошкоджений JSON — збережено резервну копію у {backup}")
                    data = {}

            data[self._round] = self._round_log
            TIMESTAMPS_FILE.write_text(json.dumps(data, indent=2))
            self._timestamps_saved = True
            print(
                f"\n✅ Timestamps збережено у {TIMESTAMPS_FILE}  (round={self._round})"
            )
            for name, ts in self._round_log.items():
                print(
                    f"   {name}: {ts['start_str']}  unix={ts['start_unix']:.0f}–{ts['end_unix']:.0f}"
                )
        except Exception as e:
            print(f"\n❌ Не вдалося зберегти timestamps: {e}")
            print(f"   Дані раунду {self._round}:")
            print(json.dumps(self._round_log, indent=2))
