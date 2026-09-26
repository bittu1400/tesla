"""Train a VAE on recorded driving frames.

Without --dr: an ordinary VAE (input = target = clean frame).
With --dr:    a denoising VAE: input is a domain-randomized copy, target the
              clean frame, so the encoder learns to ignore lighting and noise.

Writes to --out-dir: vae.pth (full model), encoder.npz (encoder weights for
numpy/the car), recon.png (rows: input, target, reconstruction), history.json.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

from car.policy import preprocess
from collect.dataset import load_frame, read_labels
from envs.augment import DomainRandomizer
from learn.nets import ConvVAE, encoder_arrays, save_npz, save_vae

MIN_FRAMES = 500


class DriveFrames(Dataset):
    def __init__(self, paths, dr: bool, seed: int = 0):
        self.paths = list(paths)
        self.dr = dr
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frame = load_frame(self.paths[idx])
        target = torch.from_numpy(preprocess(frame))
        if not self.dr:
            return target, target
        # A new DomainRandomizer samples fresh settings: one randomization per sample.
        return torch.from_numpy(preprocess(DomainRandomizer(self.rng)(frame))), target


def seed_worker(worker_id: int) -> None:
    """Each DataLoader worker gets a copy of the dataset; give each its own
    RNG (derived from torch's per-worker seed) so workers don't repeat augmentations."""
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed())


def vae_loss(recon, target, mu, logvar, beta: float = 1.0) -> torch.Tensor:
    """Per-sample summed squared error + beta * KL divergence, averaged over the batch."""
    recon_loss = F.mse_loss(recon, target, reduction="sum") / target.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return recon_loss + beta * kld


def run_epoch(model, loader, device, optimizer=None, beta: float = 1.0) -> float:
    model.train(optimizer is not None)
    total, batches = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            recon, mu, logvar = model(inp)
            loss = vae_loss(recon, target, mu, logvar, beta)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            batches += 1
    return total / max(batches, 1)


def train(model, train_loader, val_loader, epochs: int, device: str = "cpu", lr: float = 1e-3, beta: float = 1.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer, beta)
        val_loss = run_epoch(model, val_loader, device, None, beta)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"epoch {epoch + 1}/{epochs}  train {train_loss:.2f}  val {val_loss:.2f}")
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_reconstructions(model, dataset, out_path, n: int = 8) -> None:
    model.eval()
    pairs = [dataset[i] for i in range(min(n, len(dataset)))]
    inp = torch.stack([p[0] for p in pairs])
    target = torch.stack([p[1] for p in pairs])
    device = next(model.parameters()).device
    with torch.no_grad():
        recon = model.decode(model.encoder(inp.to(device))).cpu()
    rows = [torch.cat(list(images), dim=2) for images in (inp, target, recon)]  # each (3, 80, 80n)
    image = (torch.cat(rows, dim=1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", nargs="+", default=["data/drive"], help="one or more dataset dirs")
    parser.add_argument("--out-dir", default="models/vae_clean")
    parser.add_argument("--dr", action="store_true", help="denoising VAE on domain-randomized inputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = [row["frame"] for d in args.data_dir for row in read_labels(d)]
    if len(paths) < MIN_FRAMES:
        raise SystemExit(f"only {len(paths)} frames in {args.data_dir}; collect at least {MIN_FRAMES}")

    torch.manual_seed(args.seed)
    order = np.random.default_rng(args.seed).permutation(len(paths))
    n_val = max(1, int(len(paths) * args.val_fraction))
    val_set = DriveFrames([paths[i] for i in order[:n_val]], args.dr, seed=args.seed + 1)
    train_set = DriveFrames([paths[i] for i in order[n_val:]], args.dr, seed=args.seed)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, **loader_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(train_set)} train / {len(val_set)} val frames on {device}, dr={args.dr}")
    model = ConvVAE()
    history = train(model, train_loader, val_loader, args.epochs, device, beta=args.beta)

    out_dir = Path(args.out_dir)
    model.cpu()
    save_vae(model, out_dir / "vae.pth")
    save_npz(out_dir / "encoder.npz", encoder_arrays(model.encoder))
    save_reconstructions(model, val_set, out_dir / "recon.png")
    (out_dir / "history.json").write_text(json.dumps({**history, "dr": args.dr, "frames": len(paths)}, indent=2))
    print(f"best val loss {min(history['val']):.2f}; wrote {out_dir}/vae.pth, encoder.npz, recon.png")


if __name__ == "__main__":
    main()
