"""Numpy-only inference, shared by the simulator runs, the benchmark and the real car.

The Raspberry Pi runs this exact file and has no torch, so it imports only
numpy. learn/nets.py trains the same networks in torch and exports their
weights into the .npz format read here; tests check both give the same output.
"""
import numpy as np

CAMERA_SHAPE = (120, 160, 3)  # donkeycar and gym-donkeycar camera frames (H, W, C), RGB
CROP_TOP = 40  # rows of sky/background removed
CLIP_MEAN = 2.0  # SB3 SAC (gSDE) clips the actor mean to [-2, 2] before tanh
CONV_LAYERS = (0, 2, 4)  # indices of the Conv2d layers in learn.nets.Encoder.conv


def preprocess(frame) -> np.ndarray:
    """RGB uint8 120x160x3 -> float32 3x80x80 in [0, 1].

    Drops the top 40 rows, then halves the width by averaging column pairs.
    This is the only preprocessing in the project: sim, training and the car
    all call it.
    """
    frame = np.asarray(frame)
    if frame.shape != CAMERA_SHAPE:
        raise ValueError(f"expected a {CAMERA_SHAPE} RGB frame, got shape {frame.shape}")
    x = frame[CROP_TOP:].astype(np.float32) / 255.0  # 80 x 160 x 3
    x = (x[:, 0::2] + x[:, 1::2]) / 2.0  # 80 x 80 x 3
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 2, padding: int = 1) -> np.ndarray:
    """Same result as torch.nn.functional.conv2d for a single CxHxW input."""
    c_out, c_in, kh, kw = weight.shape
    x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]
    h_out, w_out = windows.shape[1], windows.shape[2]
    patches = windows.transpose(1, 2, 0, 3, 4).reshape(h_out * w_out, c_in * kh * kw)
    out = patches @ weight.reshape(c_out, -1).T + bias
    return out.T.reshape(c_out, h_out, w_out)


def load_arrays(path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _subset(arrays: dict, prefix: str) -> dict:
    return {k[len(prefix):]: np.asarray(v, dtype=np.float32) for k, v in arrays.items() if k.startswith(prefix)}


class Encoder:
    """VAE encoder mean: 3x80x80 -> latent_dim."""

    def __init__(self, arrays: dict):
        self.p = _subset(arrays, "encoder.")
        if "fc_mu.weight" not in self.p:
            raise ValueError("no encoder weights (expected keys starting with 'encoder.')")
        self.latent_dim = self.p["fc_mu.weight"].shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i in CONV_LAYERS:
            h = np.maximum(conv2d(h, self.p[f"conv.{i}.weight"], self.p[f"conv.{i}.bias"]), 0.0)
        return self.p["fc_mu.weight"] @ h.reshape(-1) + self.p["fc_mu.bias"]


class Head:
    """Deterministic steering head: [latent, prev_steer] -> steering in [-1, 1]."""

    def __init__(self, arrays: dict):
        self.p = _subset(arrays, "head.")
        if "mu.weight" not in self.p:
            raise ValueError("no head weights (expected keys starting with 'head.')")

    def __call__(self, obs: np.ndarray) -> float:
        h = np.maximum(self.p["latent_pi.0.weight"] @ obs + self.p["latent_pi.0.bias"], 0.0)
        h = np.maximum(self.p["latent_pi.2.weight"] @ h + self.p["latent_pi.2.bias"], 0.0)
        mean = self.p["mu.weight"] @ h + self.p["mu.bias"]
        return float(np.tanh(np.clip(mean, -CLIP_MEAN, CLIP_MEAN))[0])


class Policy:
    """A trained model file (encoder + head) that maps a camera frame to steering."""

    def __init__(self, path):
        arrays = load_arrays(path)
        self.encoder = Encoder(arrays)
        self.head = Head(arrays)

    def act(self, frame, prev_steer: float) -> float:
        obs = np.append(self.encoder(preprocess(frame)), np.float32(prev_steer)).astype(np.float32)
        return self.head(obs)
