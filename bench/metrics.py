# bench/metrics.py
"""Metrics computed the same way for simulator episodes and real-car trials."""
import csv
import math

import numpy as np

OSC_WINDOW = 21  # samples in the moving average (about 1 s at 20 Hz)
OSC_DEADBAND = 0.02  # steering wiggles smaller than this are ignored
EARLY_SECONDS = 5.0


def steering_jerk(steer, t) -> float:
    """Mean |d steering / dt| in 1/s."""
    s, t = np.asarray(steer, dtype=float), np.asarray(t, dtype=float)
    if len(s) < 2:
        return math.nan
    dt = np.diff(t)
    ok = dt > 0
    return float(np.mean(np.abs(np.diff(s)[ok] / dt[ok]))) if ok.any() else math.nan


def oscillation_counts(steer, t, window: int = OSC_WINDOW, deadband: float = OSC_DEADBAND) -> tuple[int, float]:
    """(sign changes of steering minus its moving average, duration they span)."""
    s, t = np.asarray(steer, dtype=float), np.asarray(t, dtype=float)
    if len(s) < window:
        return 0, 0.0
    half = window // 2
    detrended = (s - np.convolve(s, np.ones(window) / window, mode="same"))[half : len(s) - half]
    signs = np.sign(detrended[np.abs(detrended) >= deadband])
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return changes, float(t[len(t) - 1 - half] - t[half])


def oscillation_hz(steer, t, window: int = OSC_WINDOW, deadband: float = OSC_DEADBAND) -> float:
    """Weaving frequency: two sign changes around the moving average per cycle."""
    changes, duration = oscillation_counts(steer, t, window, deadband)
    return changes / 2 / duration if duration > 0 else math.nan


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate; sensible for small n."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _runs(mask) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    runs, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def parse_trial_log(path) -> dict:
    """Summarise one real-car drive log written by car/pilot_part.py."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time"]) for r in rows])
    pilot = np.array([r["mode"] != "user" for r in rows], dtype=bool)
    steer = np.array([float(r["pilot_steer"]) for r in rows])
    latency = np.array([float(r["latency_ms"]) for r in rows])
    if not pilot.any():
        return {"takeover_times": [], "pilot_start": math.nan, "jerk": math.nan, "osc_hz": math.nan,
                "latency_ms_mean": math.nan, "latency_ms_p95": math.nan}

    takeover_times = [float(t[i]) for i in range(1, len(rows)) if pilot[i - 1] and not pilot[i]]
    rates, changes, duration = [], 0, 0.0
    for a, b in _runs(pilot):  # metrics only inside autopilot stretches
        seg_t, seg_s = t[a:b], steer[a:b]
        dt = np.diff(seg_t)
        ok = dt > 0
        rates.extend(np.abs(np.diff(seg_s)[ok] / dt[ok]))
        c, d = oscillation_counts(seg_s, seg_t)
        changes, duration = changes + c, duration + d
    return {
        "takeover_times": takeover_times,
        "pilot_start": float(t[np.argmax(pilot)]),
        "jerk": float(np.mean(rates)) if rates else math.nan,
        "osc_hz": changes / 2 / duration if duration > 0 else math.nan,
        "latency_ms_mean": float(latency[pilot].mean()),
        "latency_ms_p95": float(np.percentile(latency[pilot], 95)),
    }


def trial_outcome(log: dict, completed: bool) -> dict:
    """A completed lap ends with the operator's takeover at the finish line;
    every other takeover was an intervention."""
    times = log["takeover_times"]
    interventions = times[:-1] if completed and times else times
    clean = bool(completed and not interventions)
    return {
        "completed": bool(completed),
        "interventions": len(interventions),
        "clean_lap": clean,
        "lap_time": times[-1] - log["pilot_start"] if clean and times else math.nan,
        "early_intervention": any(x - log["pilot_start"] <= EARLY_SECONDS for x in interventions),
    }
