import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import SAC

from car.policy import Head, load_arrays
from envs.sim_process import SimDisconnectedError
from learn.nets import Encoder, encoder_arrays
from learn.sac_config import SAC_CONFIG
from learn.train_sac import export_sac_policy, latest_checkpoint, train

TINY_CONFIG = {
    **SAC_CONFIG,
    "buffer_size": 1000,
    "batch_size": 16,
    "learning_starts": 10,
    "policy_kwargs": {"log_std_init": -2, "net_arch": [16, 16]},
}


class FakeLatentEnv(gym.Env):
    """33-dim observations, 5-step episodes; optionally raises
    SimDisconnectedError on a given total step to simulate the sim dying."""

    def __init__(self, crash_at_step=None):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.crash_at_step = crash_at_step
        self.total_steps = 0
        self.episode_steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        return np.zeros(33, dtype=np.float32), {}

    def step(self, action):
        self.total_steps += 1
        self.episode_steps += 1
        if self.total_steps == self.crash_at_step:
            raise SimDisconnectedError("fake crash")
        obs = np.random.standard_normal(33).astype(np.float32)
        return obs, float(-abs(action[0])), False, self.episode_steps >= 5, {}

    def close(self):
        self.closed = True


def test_latest_checkpoint_picks_highest_steps(tmp_path):
    for steps in (10, 200, 30):
        (tmp_path / f"sac_{steps}_steps.zip").touch()
    (tmp_path / "sac_replay_buffer_200_steps.pkl").touch()
    model_path, buffer_path = latest_checkpoint(tmp_path)
    assert model_path.name == "sac_200_steps.zip"
    assert buffer_path.name == "sac_replay_buffer_200_steps.pkl"


def test_latest_checkpoint_without_buffer_or_files(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    (tmp_path / "sac_50_steps.zip").touch()
    assert latest_checkpoint(tmp_path)[1] is None


def test_train_runs_to_total_timesteps(tmp_path):
    envs = []

    def make_env():
        envs.append(FakeLatentEnv())
        return envs[-1]

    model = train(make_env, total_timesteps=40, run_dir=tmp_path, checkpoint_freq=10, config=TINY_CONFIG, tensorboard=False)
    assert model.num_timesteps >= 40
    assert latest_checkpoint(tmp_path / "checkpoints") is not None
    assert envs[-1].closed


def test_train_passes_seed_to_sac(tmp_path):
    model = train(lambda: FakeLatentEnv(), total_timesteps=10, run_dir=tmp_path, checkpoint_freq=10,
                  config=TINY_CONFIG, tensorboard=False, seed=123)
    assert model.seed == 123


def test_train_resumes_from_checkpoint_after_sim_crash(tmp_path):
    envs = []

    def make_env():
        envs.append(FakeLatentEnv(crash_at_step=33 if not envs else None))
        return envs[-1]

    model = train(make_env, total_timesteps=60, run_dir=tmp_path, checkpoint_freq=10, config=TINY_CONFIG, tensorboard=False)
    assert len(envs) == 2
    assert envs[0].closed
    assert model.num_timesteps >= 60
    assert envs[1].total_steps <= 35  # resumed at the 30-step checkpoint


def test_exported_head_matches_sac_predict(tmp_path):
    model = SAC(env=FakeLatentEnv(), verbose=0, **TINY_CONFIG)
    model.learn(40)
    path = tmp_path / "policy.npz"
    export_sac_policy(model, encoder_arrays(Encoder()), path)
    arrays = load_arrays(path)
    assert any(k.startswith("encoder.") for k in arrays)
    head = Head(arrays)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = (rng.standard_normal(33) * 3).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        assert head(obs) == pytest.approx(float(action[0]), abs=1e-5)
