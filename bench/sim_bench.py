"""Simulator robustness benchmark: every model x condition x episode, one lap each.

One simulator session serves all runs: between episodes the throttle, steering
delay, encoder and image perturbation are swapped in place. Runs are
interleaved (per condition, per episode index, all models) and every model
gets the same random perturbation for a given (condition, episode).
Each finished episode is appended to --out immediately; re-running skips
finished episodes, so a crash or Ctrl+C loses at most one episode.

Model-quality gate first:  python -m bench.sim_bench --conditions nominal
"""
import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np

from bench.conditions import BY_NAME, CONDITIONS, MODELS, Condition
from bench.metrics import oscillation_hz, steering_jerk
from car.policy import Policy
from envs.augment import make_augmenter
from envs.cli import add_env_args, env_kwargs_from_args
from envs.donkey_env import SETTLE_STEPS, DonkeyLaneEnv
from envs.sim_process import SimDisconnectedError
from envs.wrappers import ActuatorLag, LatentEnv

OFFSET_STEER = 0.6


def _abs_cte(info: dict, step: int):
    """abs(cte), or None during the first SETTLE_STEPS steps after a reset,
    where the sim can report CTE spikes (DonkeyLaneEnv ignores them too)."""
    return None if step <= SETTLE_STEPS else abs(info.get("cte", 0.0))


def run_episode(head, env, condition: Condition, cte_max: float, direction: float = 1.0, laps: int = 1,
                max_offset_steps: int = 30, clock=time.monotonic) -> dict:
    obs, info = env.reset()
    forcing = condition.offset_start
    handover, recovery = None, None
    steers, times, abs_ctes = [], [], []
    step = 0
    while True:
        cte = _abs_cte(info, step)
        if forcing and (step >= max_offset_steps or (cte is not None and cte >= 0.5 * cte_max)):
            forcing, handover = False, step
        if forcing:
            steer = OFFSET_STEER * direction
        else:
            steer = head(obs)
            steers.append(steer)
            times.append(clock())
        obs, _, terminated, truncated, info = env.step(np.array([steer], dtype=np.float32))
        step += 1
        cte = _abs_cte(info, step)
        if cte is not None:
            abs_ctes.append(cte)
            if handover is not None and recovery is None and cte < 0.2 * cte_max:
                recovery = step - handover
        if info.get("lap_count", 0) >= laps:
            outcome = "success"
        elif terminated:
            outcome = "crash"
        elif truncated:
            outcome = "timeout"
        else:
            continue
        return {
            "outcome": outcome,
            "steps": step,
            "avg_abs_cte": float(np.mean(abs_ctes)) if abs_ctes else math.nan,
            "max_abs_cte": float(np.max(abs_ctes)) if abs_ctes else math.nan,
            "jerk": steering_jerk(steers, times),
            "osc_hz": oscillation_hz(steers, times),
            "lap_time": info.get("last_lap_time") if outcome == "success" else None,
            "recovery_steps": recovery,
        }


def plan_runs(models, conditions, n_episodes: int):
    return [(m, c, e) for c in conditions for e in range(n_episodes) for m in models]


def _key(model: str, condition: str, episode: int) -> tuple:
    return model, condition, int(episode)


def _done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated trailing line from a process killed mid-write
        done.add(_key(r["model"], r["condition"], r["episode"]))
    return done


def run_bench(env_factory, policies: dict, conditions, n_episodes: int, out_path, base_throttle: float,
              cte_max: float, laps: int = 1, max_restarts: int = 5, clock=time.monotonic) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out_path)
    todo = [r for r in plan_runs(list(policies), conditions, n_episodes) if _key(r[0], r[1].name, r[2]) not in done]
    stack, restarts, finished = None, 0, 0
    try:
        while finished < len(todo):
            model, condition, episode = todo[finished]
            try:
                if stack is None:
                    stack = env_factory()
                raw, lag, top = stack
                raw.throttle = base_throttle * condition.throttle_scale
                lag.delay_steps = condition.steer_delay
                top.encoder = policies[model].encoder
                rng = np.random.default_rng([zlib.crc32(condition.name.encode()), episode])
                top.augmenter = make_augmenter(condition.augmenter, rng)
                direction = 1.0 if episode % 2 == 0 else -1.0
                result = run_episode(policies[model].head, top, condition, cte_max, direction, laps, clock=clock)
            except SimDisconnectedError as exc:
                if stack is not None:
                    stack[2].close()
                    stack = None
                restarts += 1
                if restarts > max_restarts:
                    raise
                print(f"simulator disconnected ({exc}); restart {restarts}/{max_restarts}")
                continue
            with out_path.open("a") as f:
                f.write(json.dumps({"model": model, "condition": condition.name, "episode": episode, **result}) + "\n")
            finished += 1
            print(f"[{finished}/{len(todo)}] {model:9s} {condition.name:12s} ep {episode}: {result['outcome']}")
    finally:
        if stack is not None:
            stack[2].close()
    return finished


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(parser)
    parser.add_argument("--models", nargs="+", default=[f"{m}=models/{m}/policy.npz" for m in MODELS],
                        help="name=path/to/policy.npz ...")
    parser.add_argument("--conditions", nargs="+", default=[c.name for c in CONDITIONS], choices=list(BY_NAME))
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--out", default="results/sim/episodes.jsonl")
    parser.add_argument("--max-restarts", type=int, default=5)
    args = parser.parse_args()

    env_kwargs = env_kwargs_from_args(args)
    policies = {name: Policy(path) for name, path in (spec.split("=", 1) for spec in args.models)}
    first_encoder = next(iter(policies.values())).encoder

    def factory():
        raw = DonkeyLaneEnv(**env_kwargs)
        lag = ActuatorLag(raw)
        return raw, lag, LatentEnv(lag, first_encoder)

    n = run_bench(factory, policies, [BY_NAME[c] for c in args.conditions], args.n_episodes, args.out,
                  base_throttle=env_kwargs["throttle"], cte_max=env_kwargs["cte_max"], laps=args.laps,
                  max_restarts=args.max_restarts)
    print(f"ran {n} episodes -> {args.out}  (summary: python -m bench.report)")


if __name__ == "__main__":
    main()
