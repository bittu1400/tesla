import numpy as np
import pytest

from car.policy import CAMERA_SHAPE
from envs.augment import AUGMENTERS, DomainRandomizer, Distractors, LowLight, MovingShadow, make_augmenter


def _frame(value=150):
    return np.full(CAMERA_SHAPE, value, dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(AUGMENTERS))
def test_shape_dtype_and_input_untouched(name):
    frame = np.random.default_rng(0).integers(0, 256, size=CAMERA_SHAPE, dtype=np.uint8)
    original = frame.copy()
    out = make_augmenter(name, np.random.default_rng(1))(frame, 3)
    assert out.shape == CAMERA_SHAPE and out.dtype == np.uint8
    assert np.array_equal(frame, original)


def test_make_augmenter_none_and_unknown():
    assert make_augmenter(None, np.random.default_rng(0)) is None
    with pytest.raises(KeyError):
        make_augmenter("fog", np.random.default_rng(0))


def test_domain_randomizer_is_deterministic_for_a_seed():
    a = DomainRandomizer(np.random.default_rng(7))(_frame())
    b = DomainRandomizer(np.random.default_rng(7))(_frame())
    assert np.array_equal(a, b)


def test_domain_randomizer_changes_between_episodes():
    dr = DomainRandomizer(np.random.default_rng(0))
    first = dr(_frame()).astype(int)
    dr.new_episode()
    assert np.abs(dr(_frame()).astype(int) - first).mean() > 1.0


def test_training_brightness_never_reaches_low_light_strength():
    dr = DomainRandomizer(np.random.default_rng(0))
    values = []
    for _ in range(500):
        dr.new_episode()
        values.append(dr.brightness)
    assert min(values) >= 0.6 > LowLight.BRIGHTNESS


def test_low_light_darkens():
    out = LowLight(np.random.default_rng(0))(_frame(150))
    assert out.mean() < 0.5 * 150


def test_moving_shadow_moves_over_time():
    shadow = MovingShadow(np.random.default_rng(0))
    darkened = [(shadow(_frame(200), t) < 200).sum() for t in range(0, 120, 4)]
    assert max(darkened) > 0
    assert not np.array_equal(shadow(_frame(200), 0), shadow(_frame(200), 10))


def test_distractors_fixed_within_episode_and_change_between():
    d = Distractors(np.random.default_rng(0))
    first = d(_frame(), 0)
    assert np.array_equal(first, d(_frame(), 50))
    assert (first != 150).any()
    d.new_episode()
    assert not np.array_equal(first, d(_frame(), 0))
