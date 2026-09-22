import time

import numpy as np
import pytest

import envs.donkey_env as donkey_env
from envs.donkey_env import DonkeyLaneEnv
from envs.reward import CRASH_PENALTY
from envs.sim_process import SimDisconnectedError


class FakeUnderlyingEnv:
    """Stands in for the gym_donkeycar env: 5-tuple step, hit is "none" when clean."""

    def __init__(self):
        self.infos = []
        self.last_action = None
        self.closed = False
        self.hang = False

    def reset(self):
        return np.zeros((120, 160, 3), dtype=np.uint8), {"cte": 0.0, "hit": "none"}

    def step(self, action):
        if self.hang:
            time.sleep(60)
        self.last_action = np.asarray(action)
        info = self.infos.pop(0) if self.infos else {"cte": 0.0, "hit": "none", "forward_vel": 1.0}
        done = info.pop("_done", False)
        return np.zeros((120, 160, 3), dtype=np.uint8), 0.0, done, False, info

    def close(self):
        self.closed = True


@pytest.fixture
def fake():
    return FakeUnderlyingEnv()


@pytest.fixture
def captured():
    return {}


@pytest.fixture
def make_env(monkeypatch, fake, captured):
    def factory(env_name, conf):
        captured.update(env_name=env_name, conf=conf)
        return fake

    monkeypatch.setattr(donkey_env, "_make_underlying_env", factory)
    return lambda **kwargs: DonkeyLaneEnv(exe_path="remote", **kwargs)


def test_action_space_is_steering_only(make_env):
    assert make_env().action_space.shape == (1,)


def test_missing_binary_fails_fast():
    with pytest.raises(FileNotFoundError):
        DonkeyLaneEnv(exe_path="/definitely/not/here")


def test_reset_returns_raw_frame(make_env):
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (120, 160, 3) and obs.dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_step_scales_steering_and_sends_fixed_throttle(make_env, fake):
    env = make_env(steer_limit=0.8, throttle=0.3)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(np.array([0.5]))
    assert np.allclose(fake.last_action, [0.4, 0.3])
    assert reward == pytest.approx(1.0)
    assert not terminated and not truncated


def test_steering_is_clipped(make_env, fake):
    env = make_env()
    env.reset()
    env.step(np.array([5.0]))
    assert fake.last_action[0] == pytest.approx(1.0)


def test_throttle_attribute_can_change_between_episodes(make_env, fake):
    env = make_env(throttle=0.3)
    env.throttle = 0.1
    env.reset()
    env.step(np.zeros(1))
    assert fake.last_action[1] == pytest.approx(0.1)


def test_hit_none_string_does_not_terminate(make_env, fake):
    env = make_env()
    env.reset()
    fake.infos = [{"cte": 0.0, "hit": "none", "forward_vel": 1.0}]
    assert env.step(np.zeros(1))[2] is False


def test_collision_terminates(make_env, fake):
    env = make_env()
    env.reset()
    fake.infos = [{"cte": 0.0, "hit": "wall", "forward_vel": 1.0}]
    _, reward, terminated, truncated, _ = env.step(np.zeros(1))
    assert terminated is True and truncated is False
    assert reward == CRASH_PENALTY


def test_sim_game_over_terminates(make_env, fake):
    env = make_env()
    env.reset()
    fake.infos = [{"cte": 2.5, "hit": "none", "forward_vel": 1.0, "_done": True}]
    assert env.step(np.zeros(1))[2] is True


def test_truncates_at_max_episode_steps(make_env):
    env = make_env(max_episode_steps=3)
    env.reset()
    assert [env.step(np.zeros(1))[3] for _ in range(3)] == [False, False, True]
    env.reset()
    assert env.step(np.zeros(1))[3] is False


def test_hung_sim_raises_disconnected(make_env, fake):
    env = make_env(step_timeout=0.2)
    env.reset()
    fake.hang = True
    with pytest.raises(SimDisconnectedError):
        env.step(np.zeros(1))


def test_connection_failure_becomes_disconnected_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise Exception("Could not connect to server. Is it running?")

    monkeypatch.setattr(donkey_env, "_make_underlying_env", refuse)
    with pytest.raises(SimDisconnectedError, match="could not connect"):
        DonkeyLaneEnv(exe_path="remote")


def test_close_is_idempotent(make_env, fake):
    env = make_env()
    env.close()
    env.close()
    assert fake.closed


def test_cam_fov_is_sent_only_when_set(make_env, captured):
    make_env()
    assert "cam_config" not in captured["conf"]
    make_env(cam_fov=49)
    assert captured["conf"]["cam_config"] == {"img_w": 160, "img_h": 120, "fov": 49}
