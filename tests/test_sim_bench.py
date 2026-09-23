import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.conditions import BY_NAME
from bench.sim_bench import _done, plan_runs, run_bench, run_episode
from envs.sim_process import SimDisconnectedError


class ScriptedEnv:
    """Replays (info, terminated, truncated) per step; records actions."""

    def __init__(self, steps):
        self.steps = steps
        self.actions = []

    def reset(self):
        self.i = 0
        return np.zeros(33, dtype=np.float32), {"cte": 0.0, "lap_count": 0}

    def step(self, action):
        self.actions.append(float(action[0]))
        info, terminated, truncated = self.steps[self.i]
        self.i += 1
        return np.zeros(33, dtype=np.float32), 0.0, terminated, truncated, info


class CountingHead:
    def __init__(self, steer=0.2):
        self.calls = 0
        self.steer = steer

    def __call__(self, obs):
        self.calls += 1
        return self.steer


def clock():
    state = {"t": 0.0}

    def tick():
        state["t"] += 0.05
        return state["t"]

    return tick


def test_success_episode_metrics():
    env = ScriptedEnv([({"cte": 0.2, "lap_count": 0}, False, False),
                       ({"cte": -0.4, "lap_count": 1, "last_lap_time": 12.5}, False, False)])
    result = run_episode(CountingHead(), env, BY_NAME["nominal"], cte_max=2.0, clock=clock())
    assert result["outcome"] == "success" and result["steps"] == 2
    assert result["avg_abs_cte"] == pytest.approx(0.3)
    assert result["max_abs_cte"] == pytest.approx(0.4)
    assert result["lap_time"] == 12.5 and result["recovery_steps"] is None


def test_crash_and_timeout():
    crash = ScriptedEnv([({"cte": 1.0, "lap_count": 0}, True, False)])
    timeout = ScriptedEnv([({"cte": 0.1, "lap_count": 0}, False, True)])
    assert run_episode(CountingHead(), crash, BY_NAME["nominal"], 2.0, clock=clock())["outcome"] == "crash"
    result = run_episode(CountingHead(), timeout, BY_NAME["nominal"], 2.0, clock=clock())
    assert result["outcome"] == "timeout" and result["lap_time"] is None


def test_offset_start_forces_then_measures_recovery():
    env = ScriptedEnv([
        ({"cte": 0.5, "lap_count": 0}, False, False),
        ({"cte": 1.2, "lap_count": 0}, False, False),  # >= 0.5 * cte_max: hand over
        ({"cte": 0.8, "lap_count": 0}, False, False),
        ({"cte": 0.3, "lap_count": 0}, False, False),  # < 0.2 * cte_max: recovered
        ({"cte": 0.1, "lap_count": 1}, False, False),
    ])
    head = CountingHead()
    result = run_episode(head, env, BY_NAME["offset_start"], cte_max=2.0, direction=-1.0, clock=clock())
    assert env.actions == pytest.approx([-0.6, -0.6, 0.2, 0.2, 0.2])
    assert head.calls == 3
    assert result["recovery_steps"] == 2
    assert result["outcome"] == "success"


def test_offset_start_ignores_post_reset_cte_spike():
    env = ScriptedEnv([
        ({"cte": 4.5, "lap_count": 0}, False, False),  # spike right after reset (> 2*cte_max): ignored
        ({"cte": 1.2, "lap_count": 0}, False, False),  # >= 0.5 * cte_max: hand over
        ({"cte": 0.3, "lap_count": 0}, False, False),  # < 0.2 * cte_max: recovered
        ({"cte": 0.1, "lap_count": 1}, False, False),
    ])
    head = CountingHead()
    result = run_episode(head, env, BY_NAME["offset_start"], cte_max=2.0, direction=-1.0, clock=clock())
    assert env.actions == pytest.approx([-0.6, -0.6, 0.2, 0.2])  # spike alone didn't end the forcing
    assert head.calls == 2
    assert result["recovery_steps"] == 1
    assert result["max_abs_cte"] == pytest.approx(1.2)  # spike excluded from the metric
    assert result["outcome"] == "success"


def test_plan_runs_interleaves_models():
    conditions = [BY_NAME["nominal"], BY_NAME["low_light"]]
    runs = [(m, c.name, e) for m, c, e in plan_runs(["a", "b"], conditions, 2)]
    assert runs[:4] == [("a", "nominal", 0), ("b", "nominal", 0), ("a", "nominal", 1), ("b", "nominal", 1)]
    assert runs[4] == ("a", "low_light", 0) and len(runs) == 8


class FakeStack:
    """Plays raw env, lag wrapper and latent env at once; every episode is a 1-step success."""

    def __init__(self, fail=False):
        self.throttle = self.delay_steps = self.encoder = self.augmenter = None
        self.fail = fail
        self.closed = False
        self.seen = []

    def reset(self):
        if self.fail:
            raise SimDisconnectedError("boom")
        self.seen.append((self.throttle, self.delay_steps, self.encoder, type(self.augmenter).__name__))
        return np.zeros(33, dtype=np.float32), {"cte": 0.0, "lap_count": 0}

    def step(self, action):
        return np.zeros(33, dtype=np.float32), 0.0, False, False, {"cte": 0.1, "lap_count": 1, "last_lap_time": 3.0}

    def close(self):
        self.closed = True


POLICIES = {
    "m1": SimpleNamespace(encoder="enc1", head=lambda obs: 0.0),
    "m2": SimpleNamespace(encoder="enc2", head=lambda obs: 0.0),
}


def test_run_bench_configures_each_episode_and_resumes(tmp_path):
    stack = FakeStack()
    out = tmp_path / "sim" / "episodes.jsonl"
    conditions = [BY_NAME["low_light"], BY_NAME["low_battery"]]
    kwargs = dict(n_episodes=1, out_path=out, base_throttle=0.25, cte_max=2.0, clock=clock())
    assert run_bench(lambda: (stack, stack, stack), POLICIES, conditions, **kwargs) == 4
    assert stack.seen == [
        (0.25, 0, "enc1", "LowLight"),
        (0.25, 0, "enc2", "LowLight"),
        (pytest.approx(0.175), 3, "enc1", "NoneType"),
        (pytest.approx(0.175), 3, "enc2", "NoneType"),
    ]
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(l["model"], l["condition"], l["episode"], l["outcome"]) for l in lines][0] == ("m1", "low_light", 0, "success")
    assert stack.closed
    assert run_bench(lambda: (stack, stack, stack), POLICIES, conditions, **kwargs) == 0
    assert len(out.read_text().splitlines()) == 4


def test_done_skips_truncated_trailing_line(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        json.dumps({"model": "m1", "condition": "nominal", "episode": 0, "outcome": "success"}) + "\n"
        + '{"model": "m1", "condition": "nominal", "episo'  # killed mid-write
    )
    assert _done(path) == {("m1", "nominal", 0)}


def test_run_bench_rebuilds_env_after_disconnect(tmp_path):
    stacks = [FakeStack(fail=True), FakeStack()]
    made = []

    def factory():
        made.append(stacks[len(made)])
        return made[-1], made[-1], made[-1]

    out = tmp_path / "episodes.jsonl"
    n = run_bench(factory, POLICIES, [BY_NAME["nominal"]], 1, out, 0.25, 2.0, clock=clock())
    assert n == 2 and len(made) == 2
    assert stacks[0].closed
