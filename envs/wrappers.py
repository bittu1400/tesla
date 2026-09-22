from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from car.policy import preprocess


class ActuatorLag(gym.Wrapper):
    """Delays each steering command by delay_steps env steps (0 = no delay).
    The simulator's stand-in for a slow servo on a low battery. Changes to
    delay_steps take effect at the next reset."""

    def __init__(self, env: gym.Env, delay_steps: int = 0):
        super().__init__(env)
        self.delay_steps = delay_steps
        self._queue = deque()

    def reset(self, **kwargs):
        self._queue = deque([np.zeros(1, dtype=np.float32)] * self.delay_steps)
        return self.env.reset(**kwargs)

    def step(self, action):
        self._queue.append(np.asarray(action, dtype=np.float32).reshape(1))
        return self.env.step(self._queue.popleft())


class LatentEnv(gym.Wrapper):
    """Observation = [encoder(preprocess(frame)), previous steering command].

    This is exactly what the real car computes (car/policy.py::Policy.act),
    so a policy trained or evaluated here sees the same inputs on the car.
    encoder: callable 3x80x80 -> (latent_dim,) with a latent_dim attribute.
    augmenter: optional envs.augment augmenter applied to each raw frame.
    Both are public and may be swapped between episodes (the benchmark does).
    """

    def __init__(self, env: gym.Env, encoder, augmenter=None):
        super().__init__(env)
        self.encoder = encoder
        self.augmenter = augmenter
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(encoder.latent_dim + 1,), dtype=np.float32)
        self._t = 0
        self._prev = 0.0

    def _observe(self, frame) -> np.ndarray:
        if self.augmenter is not None:
            frame = self.augmenter(frame, self._t)
        return np.append(self.encoder(preprocess(frame)), self._prev).astype(np.float32)

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        if self.augmenter is not None:
            self.augmenter.new_episode()
        self._t = 0
        self._prev = 0.0
        return self._observe(frame), info

    def step(self, action):
        steer = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        frame, reward, terminated, truncated, info = self.env.step(np.array([steer], dtype=np.float32))
        self._t += 1
        self._prev = steer
        return self._observe(frame), reward, terminated, truncated, info
