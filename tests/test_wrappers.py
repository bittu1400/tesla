import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from car.policy import CAMERA_SHAPE
from envs.wrappers import ActuatorLag, LatentEnv


class FrameEnv(gym.Env):
    """Frames whose pixel value is the step count; records actions received."""

    def __init__(self):
        self.observation_space = spaces.Box(0, 255, shape=CAMERA_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.actions = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(CAMERA_SHAPE, dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(float(np.asarray(action).reshape(-1)[0]))
        self.steps += 1
        return np.full(CAMERA_SHAPE, self.steps, dtype=np.uint8), 1.0, False, False, {"cte": 0.1}


class MeanEncoder:
    """latent = [mean pixel value (0-1), 0, 0, 0]"""

    latent_dim = 4

    def __call__(self, x):
        return np.array([x.mean(), 0.0, 0.0, 0.0], dtype=np.float32)


class RecordingAugmenter:
    def __init__(self):
        self.episodes = 0
        self.ts = []

    def new_episode(self):
        self.episodes += 1

    def __call__(self, frame, t=0):
        self.ts.append(t)
        return np.full_like(frame, 255)


def test_actuator_lag_delays_steering():
    inner = FrameEnv()
    env = ActuatorLag(inner, delay_steps=2)
    env.reset()
    for a in (0.1, 0.2, 0.3, 0.4):
        env.step(np.array([a], dtype=np.float32))
    assert inner.actions == pytest.approx([0.0, 0.0, 0.1, 0.2])


def test_actuator_lag_zero_is_passthrough_and_delay_applies_on_reset():
    inner = FrameEnv()
    env = ActuatorLag(inner)
    env.reset()
    env.step(np.array([0.5], dtype=np.float32))
    env.delay_steps = 1
    env.reset()
    env.step(np.array([0.7], dtype=np.float32))
    assert inner.actions == pytest.approx([0.5, 0.0])


def test_latent_env_observation_is_latent_plus_previous_steer():
    env = LatentEnv(FrameEnv(), MeanEncoder())
    assert env.observation_space.shape == (5,)
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    assert obs[-1] == 0.0
    obs, reward, terminated, truncated, info = env.step(np.array([3.0]))
    assert obs[-1] == pytest.approx(1.0)  # clipped command
    assert obs[0] == pytest.approx(1 / 255, abs=1e-6)  # frame after step 1 has pixel value 1
    assert reward == 1.0 and info == {"cte": 0.1}
    obs, _ = env.reset()
    assert obs[-1] == 0.0


def test_latent_env_sends_clipped_one_element_action():
    inner = FrameEnv()
    env = LatentEnv(inner, MeanEncoder())
    env.reset()
    env.step(np.array([[-4.0]]))
    assert inner.actions == [-1.0]


def test_latent_env_applies_augmenter_with_step_index():
    aug = RecordingAugmenter()
    env = LatentEnv(FrameEnv(), MeanEncoder(), augmenter=aug)
    obs, _ = env.reset()
    env.step(np.zeros(1))
    env.step(np.zeros(1))
    assert aug.episodes == 1
    assert aug.ts == [0, 1, 2]
    assert obs[0] == pytest.approx(1.0)  # augmenter made the frame white
