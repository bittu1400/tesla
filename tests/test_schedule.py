from collections import Counter

from bench.conditions import BY_NAME, MODELS
from bench.schedule import BLOCK_ORDER, make_schedule
from car.trial import INSTRUCTIONS


def test_schedule_is_balanced_and_blocked():
    rows = make_schedule()
    assert len(rows) == 120
    assert [r["trial_id"] for r in rows] == [f"{i:03d}" for i in range(1, 121)]
    assert [rows[i * 20]["condition"] for i in range(6)] == list(BLOCK_ORDER)
    counts = Counter((r["condition"], r["model"]) for r in rows)
    assert set(counts.values()) == {5} and len(counts) == 24
    for condition in BLOCK_ORDER:
        for rnd in range(1, 6):
            models = [r["model"] for r in rows if r["condition"] == condition and r["round"] == rnd]
            assert sorted(models) == sorted(MODELS)


def test_schedule_is_seeded():
    assert make_schedule(seed=0) == make_schedule(seed=0)
    assert make_schedule(seed=0) != make_schedule(seed=1)


def test_every_condition_has_operator_instructions():
    assert set(INSTRUCTIONS) == set(BY_NAME) == set(BLOCK_ORDER)
