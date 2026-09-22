import math

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from car.policy import CAMERA_SHAPE
from collect.dataset import DatasetWriter, read_labels
from learn.nets import ConvVAE
from learn.train_vae import DriveFrames, save_reconstructions, train, vae_loss


def _paths(tmp_path, n=8):
    writer = DatasetWriter(tmp_path / "drive")
    rng = np.random.default_rng(0)
    for i in range(n):
        writer.add(np.full(CAMERA_SHAPE, rng.integers(50, 200), dtype=np.uint8), 0.0, 0.0, f"e{i % 2}", "scripted")
    writer.flush()
    return [r["frame"] for r in read_labels(tmp_path / "drive")]


def test_clean_dataset_input_equals_target(tmp_path):
    inp, target = DriveFrames(_paths(tmp_path), dr=False)[0]
    assert inp.shape == target.shape == (3, 80, 80)
    assert inp.dtype == torch.float32
    assert torch.equal(inp, target)


def test_dr_dataset_randomizes_input_only(tmp_path):
    paths = _paths(tmp_path)
    inp, target = DriveFrames(paths, dr=True, seed=1)[0]
    clean, _ = DriveFrames(paths, dr=False)[0]
    assert torch.equal(target, clean)
    assert not torch.equal(inp, target)


def test_vae_loss_is_finite_scalar():
    x = torch.rand(4, 3, 80, 80)
    recon, mu, logvar = ConvVAE()(x)
    loss = vae_loss(recon, x, mu, logvar)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_train_reduces_loss(tmp_path):
    torch.manual_seed(0)
    loader = DataLoader(DriveFrames(_paths(tmp_path), dr=False), batch_size=4)
    history = train(ConvVAE(latent_dim=8), loader, loader, epochs=8)
    assert len(history["train"]) == len(history["val"]) == 8
    assert all(math.isfinite(v) for v in history["train"] + history["val"])
    assert history["train"][-1] < history["train"][0]


def test_save_reconstructions_grid(tmp_path):
    out = tmp_path / "recon.png"
    save_reconstructions(ConvVAE(), DriveFrames(_paths(tmp_path, n=4), dr=True), out, n=4)
    assert cv2.imread(str(out)).shape == (240, 320, 3)
