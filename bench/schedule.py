"""Real-trial schedule: condition blocks in a fixed order (battery drains last),
5 rounds per block, all four models back-to-back in a random order each round.

    python -m bench.schedule            # writes car/schedule.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from bench.conditions import MODELS

BLOCK_ORDER = ("nominal", "distractors", "shadows", "low_light", "offset_start", "low_battery")
FIELDS = ("trial_id", "condition", "model", "round")


def make_schedule(models=MODELS, blocks=BLOCK_ORDER, rounds: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for condition in blocks:
        for rnd in range(1, rounds + 1):
            for model in rng.permutation(list(models)):
                rows.append({"trial_id": f"{len(rows) + 1:03d}", "condition": condition, "model": str(model), "round": rnd})
    return rows


def write_schedule(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="car/schedule.csv")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = make_schedule(rounds=args.rounds, seed=args.seed)
    write_schedule(rows, args.out)
    print(f"wrote {len(rows)} trials to {args.out}")


if __name__ == "__main__":
    main()
