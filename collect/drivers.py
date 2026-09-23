"""Non-learned drivers used to record training data. Steering in [-1, 1]."""
import numpy as np


class ScriptedDriver:
    """PD lane-follower on CTE that deliberately drives imperfectly.

    act() returns (executed, label):
      label    = what a clean expert would steer here:
                 clip(-cte_sign * (gain * cte + d_gain * (cte - previous cte)))
                 The derivative term stands in for heading; P alone overshoots
                 the lane centre. Call reset() at each episode start.
      executed = label + Gaussian noise, or a held hard swerve now and then
    The car follows the noisy executed steering, so the frames show
    off-centre and recovering views; BC learns the clean label for each of
    them, i.e. how to get back to the centre (noise injection with expert
    labels, as in DART).
    cte_sign flips the controller if the sim's CTE sign is the opposite of
    what we assume; calibrate it with scripts/check_sim.py.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        cte_sign: float = 1.0,
        gain: float = 0.3,
        d_gain: float = 4.0,
        noise_std: float = 0.2,
        swerve_prob: float = 0.02,
        swerve_steps: int = 15,
    ):
        self.rng = rng
        self.cte_sign = cte_sign
        self.gain = gain
        self.d_gain = d_gain
        self.noise_std = noise_std
        self.swerve_prob = swerve_prob
        self.swerve_steps = swerve_steps
        self._swerve_left = 0
        self._swerve_steer = 0.0
        self._prev_cte = None

    def reset(self) -> None:
        self._prev_cte = None

    def act(self, cte: float) -> tuple[float, float]:
        d_cte = 0.0 if self._prev_cte is None else cte - self._prev_cte
        self._prev_cte = cte
        label = float(np.clip(-self.cte_sign * (self.gain * cte + self.d_gain * d_cte), -1.0, 1.0))
        if self._swerve_left == 0 and self.rng.random() < self.swerve_prob:
            self._swerve_left = self.swerve_steps
            self._swerve_steer = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.5, 1.0))
        if self._swerve_left > 0:
            self._swerve_left -= 1
            return self._swerve_steer, label
        executed = float(np.clip(label + self.rng.normal(0.0, self.noise_std), -1.0, 1.0))
        return executed, label


def keys_to_steer(left: bool, right: bool, current: float, rate: float = 0.15) -> float:
    """Arrow keys -> steering that ramps toward the pressed direction (and back
    to 0 when released), so human labels aren't only -1 / 0 / +1."""
    target = float(right) - float(left)
    return float(np.clip(current + np.clip(target - current, -rate, rate), -1.0, 1.0))
