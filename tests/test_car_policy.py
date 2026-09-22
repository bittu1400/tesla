import numpy as np
import pytest
import torch
import torch.nn.functional as F

from car.policy import CAMERA_SHAPE, CLIP_MEAN, Head, Policy, conv2d, load_arrays, preprocess


def _frame(seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=CAMERA_SHAPE, dtype=np.uint8)


def _random_arrays(seed=0, latent_dim=32, hidden=64):
    rng = np.random.default_rng(seed)
    shapes = {
        "encoder.conv.0.weight": (32, 3, 4, 4),
        "encoder.conv.0.bias": (32,),
        "encoder.conv.2.weight": (64, 32, 4, 4),
        "encoder.conv.2.bias": (64,),
        "encoder.conv.4.weight": (128, 64, 4, 4),
        "encoder.conv.4.bias": (128,),
        "encoder.fc_mu.weight": (latent_dim, 128 * 10 * 10),
        "encoder.fc_mu.bias": (latent_dim,),
        "head.latent_pi.0.weight": (hidden, latent_dim + 1),
        "head.latent_pi.0.bias": (hidden,),
        "head.latent_pi.2.weight": (hidden, hidden),
        "head.latent_pi.2.bias": (hidden,),
        "head.mu.weight": (1, hidden),
        "head.mu.bias": (1,),
    }
    return {k: (rng.standard_normal(s) * 0.05).astype(np.float32) for k, s in shapes.items()}


def _torch_reference(arrays, frame, prev_steer):
    t = {k: torch.from_numpy(v) for k, v in arrays.items()}
    x = torch.from_numpy(preprocess(frame)).unsqueeze(0)
    for i in (0, 2, 4):
        x = F.relu(F.conv2d(x, t[f"encoder.conv.{i}.weight"], t[f"encoder.conv.{i}.bias"], stride=2, padding=1))
    latent = F.linear(x.flatten(1), t["encoder.fc_mu.weight"], t["encoder.fc_mu.bias"])
    obs = torch.cat([latent, torch.tensor([[prev_steer]])], dim=1)
    h = F.relu(F.linear(obs, t["head.latent_pi.0.weight"], t["head.latent_pi.0.bias"]))
    h = F.relu(F.linear(h, t["head.latent_pi.2.weight"], t["head.latent_pi.2.bias"]))
    mean = F.linear(h, t["head.mu.weight"], t["head.mu.bias"])
    return torch.tanh(torch.clamp(mean, -CLIP_MEAN, CLIP_MEAN)).item()


def test_preprocess_shape_dtype_range():
    x = preprocess(_frame())
    assert x.shape == (3, 80, 80)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_preprocess_drops_top_rows():
    frame = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    frame[:40] = 255
    assert preprocess(frame).max() == 0.0


def test_preprocess_halves_width_by_averaging_column_pairs():
    frame = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    frame[:, 0::2] = 200
    frame[:, 1::2] = 100
    assert np.allclose(preprocess(frame), 150 / 255, atol=1e-6)


def test_preprocess_keeps_rgb_channel_order():
    frame = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    frame[..., 0] = 255
    x = preprocess(frame)
    assert np.all(x[0] == 1.0) and np.all(x[1:] == 0.0)


def test_preprocess_rejects_other_shapes():
    with pytest.raises(ValueError, match="expected"):
        preprocess(np.zeros((240, 320, 3), dtype=np.uint8))


def test_conv2d_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.random((3, 80, 80), dtype=np.float32)
    w = rng.standard_normal((8, 3, 4, 4)).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    ours = conv2d(x, w, b)
    theirs = F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b), stride=2, padding=1)[0]
    assert ours.shape == (8, 40, 40)
    assert np.allclose(ours, theirs.numpy(), atol=1e-5)


def test_policy_matches_torch_reference(tmp_path):
    arrays = _random_arrays()
    path = tmp_path / "policy.npz"
    np.savez(path, **arrays)
    policy = Policy(path)
    assert policy.encoder.latent_dim == 32
    frame = _frame(1)
    for prev in (0.0, 0.7):
        assert policy.act(frame, prev) == pytest.approx(_torch_reference(arrays, frame, prev), abs=1e-4)


def test_head_output_is_bounded_by_clipped_tanh():
    arrays = _random_arrays()
    arrays["head.mu.bias"] = np.array([50.0], dtype=np.float32)
    assert Head(arrays)(np.zeros(33, dtype=np.float32)) == pytest.approx(np.tanh(CLIP_MEAN), abs=1e-6)


def test_load_arrays_roundtrip(tmp_path):
    arrays = _random_arrays()
    np.savez(tmp_path / "p.npz", **arrays)
    loaded = load_arrays(tmp_path / "p.npz")
    assert set(loaded) == set(arrays)
    assert np.array_equal(loaded["head.mu.weight"], arrays["head.mu.weight"])


def test_missing_encoder_weights_raise(tmp_path):
    arrays = {k: v for k, v in _random_arrays().items() if k.startswith("head.")}
    np.savez(tmp_path / "head_only.npz", **arrays)
    with pytest.raises(ValueError, match="encoder"):
        Policy(tmp_path / "head_only.npz")
