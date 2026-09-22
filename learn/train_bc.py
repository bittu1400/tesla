"""Behavioral cloning: train the policy head to copy recorded steering.

The encoder comes frozen from a trained VAE (the same encoder the matching
SAC model uses), and the head is the same PolicyHead SAC's actor exports to,
so BC and RL differ only in how the head was trained.

Inputs: [encoder latent, previous steering (+ small noise against copycat)].
Target: the steering label (expert label for scripted data, the driver's
steering for manual data). Validation is split by episode.
Writes --out (policy.npz: encoder + head) and history.json next to it.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from car.policy import preprocess
from collect.dataset import load_frame, read_labels
from envs.augment import DomainRandomizer
from learn.nets import PolicyHead, encoder_arrays, head_arrays, load_vae, save_npz
from learn.train_vae import seed_worker


class BCData(Dataset):
    def __init__(self, rows, dr: bool, seed: int = 0, prev_noise: float = 0.0):
        self.rows = list(rows)
        self.dr = dr
        self.prev_noise = prev_noise
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        frame = load_frame(row["frame"])
        if self.dr:
            frame = DomainRandomizer(self.rng)(frame)
        prev = row["prev_steer"] + (self.rng.normal(0.0, self.prev_noise) if self.prev_noise > 0 else 0.0)
        return (
            torch.from_numpy(preprocess(frame)),
            torch.tensor([prev], dtype=torch.float32),
            torch.tensor([row["steer"]], dtype=torch.float32),
        )


def split_by_episode(rows, val_fraction: float, seed: int):
    """Neighbouring frames nearly repeat each other, so whole episodes go to
    either train or validation."""
    episodes = sorted({r["episode"] for r in rows})
    if len(episodes) < 2:
        raise ValueError(f"need at least 2 episodes to split, found {len(episodes)}")
    order = np.random.default_rng(seed).permutation(len(episodes))
    n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_fraction)))
    val_episodes = {episodes[i] for i in order[:n_val]}
    return [r for r in rows if r["episode"] not in val_episodes], [r for r in rows if r["episode"] in val_episodes]


def run_epoch(encoder, head, loader, device, optimizer=None) -> float:
    head.train(optimizer is not None)
    total, batches = 0.0, 0
    for image, prev, steer in loader:
        image, prev, steer = image.to(device), prev.to(device), steer.to(device)
        with torch.no_grad():
            latent = encoder(image)
        with torch.set_grad_enabled(optimizer is not None):
            loss = F.mse_loss(head(torch.cat([latent, prev], dim=1)), steer)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


def train(encoder, head, train_loader, val_loader, epochs: int, device: str = "cpu", lr: float = 1e-3):
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        train_loss = run_epoch(encoder, head, train_loader, device, optimizer)
        val_loss = run_epoch(encoder, head, val_loader, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"epoch {epoch + 1}/{epochs}  train mse {train_loss:.4f}  val mse {val_loss:.4f}")
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(head.state_dict())
    if best_state is not None:
        head.load_state_dict(best_state)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/drive")
    parser.add_argument("--vae", required=True, help="vae.pth whose encoder this policy uses")
    parser.add_argument("--out", required=True, help="e.g. models/bc_clean/policy.npz")
    parser.add_argument("--dr", action="store_true", help="domain-randomize every training sample")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prev-noise", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_labels(args.data_dir)
    train_rows, val_rows = split_by_episode(rows, args.val_fraction, args.seed)
    torch.manual_seed(args.seed)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, worker_init_fn=seed_worker)
    train_loader = DataLoader(BCData(train_rows, args.dr, args.seed, args.prev_noise), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(BCData(val_rows, args.dr, args.seed + 1), **loader_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(train_rows)} train / {len(val_rows)} val frames on {device}, dr={args.dr}")
    encoder = load_vae(args.vae).encoder
    head = PolicyHead(obs_dim=encoder.latent_dim + 1)
    history = train(encoder, head, train_loader, val_loader, args.epochs, device, args.lr)

    save_npz(args.out, {**encoder_arrays(encoder), **head_arrays(head)})
    settings = {**vars(args), "history": history}
    Path(args.out).with_name("history.json").write_text(json.dumps(settings, indent=2))
    print(f"best val mse {min(history['val']):.4f}; wrote {args.out}")


if __name__ == "__main__":
    main()
