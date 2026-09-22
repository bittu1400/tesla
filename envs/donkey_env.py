import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from car.policy import CAMERA_SHAPE
from envs.reward import compute_reward
from envs.sim_process import SimDisconnectedError, call_with_timeout, kill_sim, launch_sim


def _make_underlying_env(env_name: str, conf: dict):
    """Thin factory around gym.make so tests can monkeypatch it."""
    import gym_donkeycar  # noqa: F401  (registers the donkey-* envs)

    return gym.make(env_name, conf=conf).unwrapped


class DonkeyLaneEnv(gym.Env):
    """Steering-only lane-following wrapper around gym-donkeycar.

    Action: [steering] in [-1, 1], scaled by steer_limit.
    Throttle: the fixed value in self.throttle, sent every step. It is a
    public attribute so the benchmark can lower it for the low_battery condition.
    Observation: the raw uint8 RGB camera frame.
    terminated: collision / off lane / sim game-over.
    truncated: max_episode_steps reached.
    Raises SimDisconnectedError if the sim won't start or stops answering;
    close() this env and build a new one.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        exe_path: str,
        env_name: str = "donkey-generated-track-v0",
        port: int = 9091,
        cte_max: float = 2.0,
        max_episode_steps: int = 2000,
        throttle: float = 0.25,
        steer_limit: float = 1.0,
        cam_fov: int = 0,
        step_timeout: float = 10.0,
        startup_timeout: float = 120.0,
    ):
        super().__init__()
        if exe_path != "remote" and not os.path.isfile(exe_path):
            raise FileNotFoundError(
                f"simulator binary not found: {exe_path!r} (pass 'remote' to use a sim you started yourself)"
            )
        self.cte_max = cte_max
        self.max_episode_steps = max_episode_steps
        self.throttle = throttle
        self.steer_limit = steer_limit
        self.step_timeout = step_timeout
        self.observation_space = spaces.Box(0, 255, shape=CAMERA_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._steps = 0
        self._underlying = None
        self._proc = None
        if exe_path != "remote":
            self._proc = launch_sim(exe_path, port, startup_timeout)
        # "remote": gym-donkeycar must not start a sim; we already did (or the user did).
        # max_cte makes the sim's own game-over check use our lane width.
        conf = {"exe_path": "remote", "port": port, "max_cte": cte_max, "start_delay": 0.0}
        if cam_fov > 0:
            conf["cam_config"] = {"img_w": CAMERA_SHAPE[1], "img_h": CAMERA_SHAPE[0], "fov": cam_fov}
        try:
            self._underlying = call_with_timeout(_make_underlying_env, startup_timeout, env_name, conf)
        except SimDisconnectedError:
            self.close()
            raise
        except Exception as exc:
            # gym-donkeycar raises a bare Exception when it can't connect.
            self.close()
            raise SimDisconnectedError(f"could not connect to simulator: {exc}") from exc

    def _guarded(self, fn, *args):
        try:
            return call_with_timeout(fn, self.step_timeout, *args)
        except (ConnectionError, OSError) as exc:
            raise SimDisconnectedError(str(exc)) from exc

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        frame, info = self._guarded(self._underlying.reset)
        self._steps = 0
        return np.asarray(frame, dtype=np.uint8), info

    def step(self, action):
        steer = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        sim_action = np.array([steer * self.steer_limit, self.throttle], dtype=np.float32)
        frame, _sim_reward, sim_done, _sim_truncated, info = self._guarded(self._underlying.step, sim_action)
        self._steps += 1
        reward, terminated = compute_reward(
            cte=info.get("cte", 0.0),
            hit=info.get("hit", "none"),
            forward_vel=info.get("forward_vel", 0.0),
            cte_max=self.cte_max,
            sim_done=sim_done,
        )
        truncated = not terminated and self._steps >= self.max_episode_steps
        return np.asarray(frame, dtype=np.uint8), reward, terminated, truncated, info

    def close(self):
        # Kill the sim first: that closes its socket, which unblocks
        # gym-donkeycar's network thread so its own close() can finish.
        kill_sim(self._proc)
        self._proc = None
        if self._underlying is not None:
            try:
                call_with_timeout(self._underlying.close, 10.0)
            except Exception:
                pass  # best effort: the sim is already gone
            self._underlying = None
