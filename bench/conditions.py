# bench/conditions.py
"""The six evaluation conditions (same names in sim and on the real car) and the four models."""
from dataclasses import dataclass

MODELS = ("sac_clean", "sac_dr", "bc_clean", "bc_dr")


@dataclass(frozen=True)
class Condition:
    name: str
    augmenter: str | None = None  # envs.augment name applied to camera frames (sim)
    steer_delay: int = 0  # steps of steering delay (sim)
    throttle_scale: float = 1.0  # multiplier on the fixed sim throttle
    offset_start: bool = False  # push the car off its line before handing over


CONDITIONS = (
    Condition("nominal"),
    Condition("low_light", augmenter="low_light"),
    Condition("shadows", augmenter="shadows"),
    Condition("distractors", augmenter="distractors"),
    Condition("low_battery", steer_delay=3, throttle_scale=0.7),
    Condition("offset_start", offset_start=True),
)
BY_NAME = {c.name: c for c in CONDITIONS}
NOISE_CONDITIONS = tuple(c.name for c in CONDITIONS if c.name != "nominal")
