import csv
import math

import numpy as np
import pytest

from bench.conditions import CONDITIONS, NOISE_CONDITIONS
from bench.metrics import oscillation_hz, parse_trial_log, steering_jerk, trial_outcome, wilson_interval
from envs.augment import AUGMENTERS

LOG_FIELDS = ("time", "mode", "pilot_steer", "user_steer", "latency_ms")


def write_log(path, modes, steer=None, dt=0.05, latency=12.0):
    steer = steer if steer is not None else [0.0] * len(modes)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_FIELDS)
        for i, (mode, s) in enumerate(zip(modes, steer)):
            writer.writerow([f"{1000 + i * dt:.3f}", mode, f"{s:.4f}", "0.0000", f"{latency:.2f}"])


def test_condition_augmenters_exist():
    assert all(c.augmenter is None or c.augmenter in AUGMENTERS for c in CONDITIONS)
    assert "nominal" not in NOISE_CONDITIONS and len(NOISE_CONDITIONS) == 5


def test_steering_jerk_of_ramp():
    t = np.arange(0, 1, 0.05)
    assert steering_jerk(0.1 * t, t) == pytest.approx(0.1)
    assert math.isnan(steering_jerk([0.2], [0.0]))


def test_oscillation_of_two_hertz_sine():
    t = np.arange(0, 10, 0.05)
    assert oscillation_hz(0.3 * np.sin(2 * np.pi * 2.0 * t), t) == pytest.approx(2.0, abs=0.2)


def test_oscillation_of_constant_and_short_signals():
    t = np.arange(0, 5, 0.05)
    assert oscillation_hz(np.full(len(t), 0.4), t) == 0.0
    assert math.isnan(oscillation_hz([0.1, 0.2], [0.0, 0.05]))


def test_wilson_interval():
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    lo, hi = wilson_interval(5, 10)
    assert lo + hi == pytest.approx(1.0)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_parse_trial_log(tmp_path):
    # 10 user, 100 pilot, 10 user (intervention), 60 pilot, 5 user (finish)
    modes = ["user"] * 10 + ["local"] * 100 + ["user"] * 10 + ["local"] * 60 + ["user"] * 5
    steer = [0.3 * math.sin(2 * math.pi * 1.0 * i * 0.05) for i in range(len(modes))]
    path = tmp_path / "trial_001.csv"
    write_log(path, modes, steer)
    log = parse_trial_log(path)
    assert log["takeover_times"] == pytest.approx([1005.5, 1009.0])
    assert log["pilot_start"] == pytest.approx(1000.5)
    assert log["latency_ms_mean"] == pytest.approx(12.0)
    assert log["osc_hz"] == pytest.approx(1.0, abs=0.25)
    assert log["jerk"] > 0
    assert log["last_mode"] == "user"


def test_parse_log_without_pilot_samples(tmp_path):
    path = tmp_path / "t.csv"
    write_log(path, ["user"] * 5)
    log = parse_trial_log(path)
    assert log["takeover_times"] == [] and math.isnan(log["jerk"])


def test_trial_outcome():
    log = {"takeover_times": [1005.5, 1009.0], "pilot_start": 1000.5, "last_mode": "user"}
    done = trial_outcome(log, completed=True)
    assert done["interventions"] == 1 and not done["clean_lap"]
    assert math.isnan(done["lap_time"])
    assert done["early_intervention"]  # 5.0 s after start
    clean = trial_outcome({"takeover_times": [1009.0], "pilot_start": 1000.5, "last_mode": "user"}, completed=True)
    assert clean["clean_lap"] and clean["lap_time"] == pytest.approx(8.5)
    assert not clean["early_intervention"]
    failed = trial_outcome(log, completed=False)
    assert failed["interventions"] == 2 and not failed["clean_lap"]


def test_trial_outcome_counts_final_takeover_when_operator_never_switched_back():
    # Ctrl+C while still in "local" mode: the last takeover was never returned,
    # so a lap that otherwise looked clean still counts its one intervention.
    log = {"takeover_times": [1005.5], "pilot_start": 1000.5, "last_mode": "local"}
    result = trial_outcome(log, completed=True)
    assert result["interventions"] == 1 and not result["clean_lap"]
