import numpy as np
import pytest

from car.policy import CAMERA_SHAPE
from collect.collect import run_scripted
from collect.dataset import DatasetWriter, load_frame, read_labels
from collect.drivers import ScriptedDriver


class FakeEnv:
    """cte is 0 after reset and 1.0 after every step; crashes every 4 steps."""

    def __init__(self):
        self.resets = 0
        self.steps = 0
        self.actions = []

    def reset(self):
        self.resets += 1
        return np.zeros(CAMERA_SHAPE, dtype=np.uint8), {"cte": 0.0}

    def step(self, action):
        self.steps += 1
        self.actions.append(float(action[0]))
        terminated = self.steps % 4 == 0
        return np.full(CAMERA_SHAPE, self.steps, dtype=np.uint8), 0.0, terminated, False, {"cte": 1.0}


def test_run_scripted_aligns_frames_labels_and_previous_steering(tmp_path):
    env = FakeEnv()
    writer = DatasetWriter(tmp_path, save_every=5)
    driver = ScriptedDriver(np.random.default_rng(0), d_gain=0.0, noise_std=0.0, swerve_prob=0.0)
    run_scripted(env, driver, writer, n_frames=10, run_id="run")
    writer.flush()
    rows = read_labels(tmp_path)
    assert len(rows) == 10
    assert env.resets == 3  # initial + crashes at steps 4 and 8
    # row 0: reset frame, cte 0
    assert rows[0]["steer"] == 0.0 and rows[0]["prev_steer"] == 0.0
    # row 1: frame after step 1 (pixel value 1), cte 1 -> label -0.3, previous executed steer 0
    assert np.array_equal(load_frame(rows[1]["frame"]), np.full(CAMERA_SHAPE, 1, dtype=np.uint8))
    assert rows[1]["steer"] == pytest.approx(-0.3) and rows[1]["prev_steer"] == 0.0
    assert rows[2]["prev_steer"] == pytest.approx(-0.3)
    # step 4 crashed: row 4 starts a new episode with prev_steer reset
    assert rows[3]["episode"] == "run-0" and rows[4]["episode"] == "run-1"
    assert rows[4]["prev_steer"] == 0.0
