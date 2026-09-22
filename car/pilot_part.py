"""donkeycar part that drives with a benchmark policy and logs every frame.

Added to the car's manage.py (see README, "Raspberry Pi setup"):
  inputs  cam/image_array, user/mode, user/angle
  outputs pilot/angle, pilot/throttle
donkeycar's DriveMode part then uses these outputs only in autopilot
(local) mode; in user mode the operator drives, and the log records it.
"""
import csv
import time
from pathlib import Path

import numpy as np

from car.policy import Policy

LOG_FIELDS = ("time", "mode", "pilot_steer", "user_steer", "latency_ms")


class RobustPilot:
    def __init__(self, policy_path, throttle: float, log_path, image_dir=None, image_every: int = 5,
                 steer_gain: float = 1.0, policy=None):
        self.policy = policy if policy is not None else Policy(policy_path)
        self.throttle = throttle
        self.steer_gain = steer_gain  # ponytail: fixed at 1.0 for every trial; a knob only if steering range differs from sim
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_every = image_every
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", newline="", buffering=1)  # line-buffered: rows survive a hard stop
        self._writer = csv.writer(self._file)
        self._writer.writerow(LOG_FIELDS)
        self._prev = 0.0
        self._frames = 0

    def run(self, img, mode, user_steer):
        if img is None:
            return 0.0, 0.0
        start = time.perf_counter()
        raw = self.policy.act(img, self._prev)
        latency_ms = (time.perf_counter() - start) * 1000.0
        steer = float(np.clip(raw * self.steer_gain, -1.0, 1.0))
        user = float(user_steer or 0.0)
        # In the sim the policy sees its own previous command; when the operator
        # drives, the car's actual previous steering is the user's.
        self._prev = raw if mode != "user" else user
        self._writer.writerow([f"{time.time():.3f}", mode, f"{steer:.4f}", f"{user:.4f}", f"{latency_ms:.2f}"])
        if self.image_dir is not None and self._frames % self.image_every == 0:
            from PIL import Image

            self.image_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(self.image_dir / f"{self._frames:06d}.jpg", quality=90)
        self._frames += 1
        return steer, self.throttle

    def shutdown(self):
        self._file.close()
