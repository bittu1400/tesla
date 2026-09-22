import numpy as np
import pytest
import torch

from car.policy import CAMERA_SHAPE, CLIP_MEAN, Policy
from learn.nets import (
    ConvVAE,
    Encoder,
    PolicyHead,
    encoder_arrays,
    frames_to_tensor,
    head_arrays,
    load_vae,
    save_npz,
    save_vae,
)


def test_encoder_output_shape():
    assert Encoder()(torch.rand(4, 3, 80, 80)).shape == (4, 32)


def test_encoder_rejects_wrong_input_size():
    with pytest.raises(ValueError, match="expected"):
        Encoder()(torch.rand(1, 3, 120, 160))


def test_vae_forward_and_decode_shapes():
    x = torch.rand(4, 3, 80, 80)
    recon, mu, logvar = ConvVAE()(x)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (4, 32)
    out = ConvVAE().decode(torch.randn(2, 32))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_policy_head_shape_and_bounds():
    out = PolicyHead()(torch.randn(5, 33) * 100)
    assert out.shape == (5, 1)
    assert out.abs().max() <= np.tanh(CLIP_MEAN) + 1e-6


def test_save_load_vae_roundtrip(tmp_path):
    model = ConvVAE(latent_dim=16)
    path = tmp_path / "sub" / "vae.pth"
    save_vae(model, path)
    loaded = load_vae(path)
    assert loaded.latent_dim == 16
    assert not loaded.training
    x = torch.rand(2, 3, 80, 80)
    assert torch.allclose(model.encoder(x), loaded.encoder(x))


def test_frames_to_tensor():
    frames = [np.zeros(CAMERA_SHAPE, dtype=np.uint8), np.full(CAMERA_SHAPE, 255, dtype=np.uint8)]
    x = frames_to_tensor(frames)
    assert x.shape == (2, 3, 80, 80)
    assert x.dtype == torch.float32
    assert x[1].min() == 1.0


def test_exported_npz_matches_torch_forward(tmp_path):
    torch.manual_seed(0)
    encoder, head = Encoder().eval(), PolicyHead().eval()
    path = tmp_path / "m" / "policy.npz"
    save_npz(path, {**encoder_arrays(encoder), **head_arrays(head)})
    frame = np.random.default_rng(0).integers(0, 256, size=CAMERA_SHAPE, dtype=np.uint8)
    prev = -0.3
    with torch.no_grad():
        latent = encoder(frames_to_tensor([frame]))
        expected = head(torch.cat([latent, torch.tensor([[prev]])], dim=1)).item()
    assert Policy(path).act(frame, prev) == pytest.approx(expected, abs=1e-4)
