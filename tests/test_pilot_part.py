import csv

import numpy as np
import pytest

from car.pilot_part import RobustPilot
from car.policy import CAMERA_SHAPE

FRAME = np.zeros(CAMERA_SHAPE, dtype=np.uint8)


class FakePolicy:
    def __init__(self, steer=0.4):
        self.steer = steer
        self.prevs = []

    def act(self, frame, prev_steer):
        self.prevs.append(prev_steer)
        return self.steer


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_run_steers_logs_and_tracks_previous_steering(tmp_path):
    policy = FakePolicy()
    pilot = RobustPilot(None, throttle=0.3, log_path=tmp_path / "log.csv", policy=policy)
    assert pilot.run(FRAME, "local", 0.0) == (pytest.approx(0.4), 0.3)
    pilot.run(FRAME, "user", -0.5)
    pilot.run(FRAME, "local", 0.0)
    pilot.shutdown()
    assert policy.prevs == pytest.approx([0.0, 0.4, -0.5])
    rows = _rows(tmp_path / "log.csv")
    assert [r["mode"] for r in rows] == ["local", "user", "local"]
    assert float(rows[1]["user_steer"]) == pytest.approx(-0.5)
    assert float(rows[0]["pilot_steer"]) == pytest.approx(0.4)
    assert float(rows[0]["latency_ms"]) >= 0.0


def test_missing_frame_is_safe(tmp_path):
    pilot = RobustPilot(None, 0.3, tmp_path / "log.csv", policy=FakePolicy())
    assert pilot.run(None, "local", 0.0) == (0.0, 0.0)
    pilot.shutdown()
    assert _rows(tmp_path / "log.csv") == []


def test_steer_gain_clips_output_but_not_previous_steering(tmp_path):
    policy = FakePolicy(0.4)
    pilot = RobustPilot(None, 0.3, tmp_path / "log.csv", steer_gain=5.0, policy=policy)
    assert pilot.run(FRAME, "local", 0.0)[0] == 1.0
    pilot.run(FRAME, "local", 0.0)
    pilot.shutdown()
    assert policy.prevs[1] == pytest.approx(0.4)


def test_saves_every_nth_frame(tmp_path):
    pilot = RobustPilot(None, 0.3, tmp_path / "log.csv", image_dir=tmp_path / "img", image_every=2, policy=FakePolicy())
    for _ in range(5):
        pilot.run(FRAME, "local", 0.0)
    pilot.shutdown()
    assert sorted(p.name for p in (tmp_path / "img").iterdir()) == ["000000.jpg", "000002.jpg", "000004.jpg"]
