CRASH_PENALTY = -10.0


def is_hit(hit) -> bool:
    """gym-donkeycar reports a collision as the name of the object hit and the
    string "none" otherwise. "none" is truthy, so never test it for truthiness."""
    return hit is not None and hit != "none"


def compute_reward(cte: float, hit, forward_vel: float, cte_max: float, sim_done: bool) -> tuple[float, bool]:
    """Returns (reward, terminated) for steering-only lane following.

    Throttle is fixed, so reward only asks: how close to the lane centre?
      reward = 1 at the centre, falling linearly to 0 at cte_max
      0 when not moving forward (after a spin the car may face backwards)
    A collision, or the sim's own game-over (|cte| > max_cte, missed
    checkpoint, disqualification), ends the episode with CRASH_PENALTY.
    Off-lane is deliberately not re-checked here: the sim ignores the CTE
    spikes it reports just after a reset, and re-checking would not.
    """
    if sim_done or is_hit(hit):
        return CRASH_PENALTY, True
    if forward_vel <= 0.0:
        return 0.0, False
    return 1.0 - min(abs(cte) / cte_max, 1.0), False
