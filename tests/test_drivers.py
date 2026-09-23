import numpy as np
import pytest

from collect.drivers import ScriptedDriver, keys_to_steer


def _driver(**kwargs):
    defaults = dict(rng=np.random.default_rng(0), noise_std=0.0, swerve_prob=0.0)
    return ScriptedDriver(**{**defaults, **kwargs})


def test_label_steers_against_cte():
    assert _driver().act(1.0)[1] == pytest.approx(-0.3)
    assert _driver().act(-1.0)[1] == pytest.approx(0.3)
    assert _driver(cte_sign=-1.0).act(1.0)[1] == pytest.approx(0.3)


def test_derivative_term_damps_approach_and_resets():
    driver = _driver(gain=0.3, d_gain=4.0)
    driver.act(-1.0)
    # moving toward the centre (cte -1.0 -> -0.9): P says +0.27, D says -0.4
    assert driver.act(-0.9)[1] == pytest.approx(0.27 - 0.4)
    driver.reset()
    assert driver.act(-0.9)[1] == pytest.approx(0.27)


def test_without_noise_executed_equals_label():
    executed, label = _driver().act(0.4)
    assert executed == label


def test_noise_changes_executed_but_not_label():
    driver = _driver(noise_std=0.3)
    pairs = [driver.act(0.4) for _ in range(20)]
    assert all(label == pytest.approx(-0.12) for _, label in pairs)
    assert any(abs(executed - label) > 0.01 for executed, label in pairs)


def test_outputs_are_clipped():
    executed, label = _driver(gain=10.0).act(5.0)
    assert executed == -1.0 and label == -1.0


def test_swerve_holds_steering_while_label_stays_expert():
    driver = _driver(swerve_prob=1.0, swerve_steps=5)
    pairs = [driver.act(0.0) for _ in range(5)]
    assert len({executed for executed, _ in pairs}) == 1
    assert abs(pairs[0][0]) >= 0.5
    assert all(label == 0.0 for _, label in pairs)


def test_keys_to_steer_ramps_and_returns_to_centre():
    s = 0.0
    for _ in range(3):
        s = keys_to_steer(False, True, s)
    assert s == pytest.approx(0.45)
    for _ in range(20):
        s = keys_to_steer(False, True, s)
    assert s == pytest.approx(1.0)
    s = keys_to_steer(False, False, s)
    assert s == pytest.approx(0.85)
    assert keys_to_steer(True, True, 0.0) == 0.0
