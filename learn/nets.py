"""Torch versions of the networks in car/policy.py, used for training.

Layer names here define the .npz keys car/policy.py reads, so rename nothing
without updating both (tests/test_nets.py checks they agree).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from car.policy import CLIP_MEAN, preprocess

LATENT_DIM = 32
HIDDEN = 64
_FLAT = 128 * 10 * 10


class Encoder(nn.Module):
    """3x80x80 frame -> latent mean. The part of the VAE every policy uses."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 80 -> 40
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 40 -> 20
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 20 -> 10
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(_FLAT, latent_dim)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (3, 80, 80):
            raise ValueError(f"expected (batch, 3, 80, 80), got {tuple(x.shape)}")
        return self.conv(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.features(x))


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.fc_logvar = nn.Linear(_FLAT, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, _FLAT)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 10 -> 20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 20 -> 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 40 -> 80
            nn.Sigmoid(),
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decode(z).view(-1, 128, 10, 10))

    def forward(self, x: torch.Tensor):
        h = self.encoder.features(x)
        mu, logvar = self.encoder.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


class PolicyHead(nn.Module):
    """Same layout and output as SB3 SAC's deterministic gSDE actor with
    net_arch=[hidden, hidden], so SAC weights copy straight in (learn/train_sac.py)
    and BC trains this exact module."""

    def __init__(self, obs_dim: int = LATENT_DIM + 1, hidden: int = HIDDEN):
        super().__init__()
        self.latent_pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.clamp(self.mu(self.latent_pi(obs)), -CLIP_MEAN, CLIP_MEAN))


def frames_to_tensor(frames) -> torch.Tensor:
    return torch.from_numpy(np.stack([preprocess(f) for f in frames]))


def save_vae(model: ConvVAE, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "latent_dim": model.latent_dim}, path)


def load_vae(path) -> ConvVAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ConvVAE(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def _arrays(module: nn.Module, prefix: str) -> dict:
    return {f"{prefix}{k}": v.detach().cpu().numpy().astype(np.float32) for k, v in module.state_dict().items()}


def encoder_arrays(encoder: Encoder) -> dict:
    return _arrays(encoder, "encoder.")


def head_arrays(head: PolicyHead) -> dict:
    return _arrays(head, "head.")


def save_npz(path, arrays: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
