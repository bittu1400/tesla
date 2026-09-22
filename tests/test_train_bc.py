import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from car.policy import CAMERA_SHAPE, Policy
from collect.dataset import DatasetWriter, read_labels
from learn.nets import Encoder, PolicyHead, encoder_arrays, head_arrays, save_npz
from learn.train_bc import BCData, split_by_episode, train


def _rows(tmp_path, n=12, episodes=3):
    writer = DatasetWriter(tmp_path / "drive")
    for i in range(n):
        value = 40 + 15 * i
        writer.add(np.full(CAMERA_SHAPE, value, dtype=np.uint8), steer=(i - n / 2) / n, prev_steer=0.1,
                   episode=f"e{i % episodes}", source="scripted")
    writer.flush()
    return read_labels(tmp_path / "drive")


def test_split_by_episode_is_disjoint_and_complete(tmp_path):
    rows = _rows(tmp_path)
    train_rows, val_rows = split_by_episode(rows, val_fraction=0.34, seed=0)
    assert len(train_rows) + len(val_rows) == len(rows)
    assert not {r["episode"] for r in train_rows} & {r["episode"] for r in val_rows}
    assert val_rows and train_rows


def test_split_needs_two_episodes(tmp_path):
    with pytest.raises(ValueError, match="episodes"):
        split_by_episode(_rows(tmp_path, episodes=1), 0.1, 0)


def test_bcdata_items(tmp_path):
    rows = _rows(tmp_path)
    image, prev, steer = BCData(rows, dr=False)[3]
    assert image.shape == (3, 80, 80) and prev.shape == steer.shape == (1,)
    assert prev.item() == pytest.approx(0.1)
    assert steer.item() == pytest.approx(rows[3]["steer"], abs=1e-4)
    noisy_prev = [BCData(rows, dr=False, seed=s, prev_noise=0.5)[3][1].item() for s in range(5)]
    assert len(set(noisy_prev)) > 1
    assert not torch.equal(BCData(rows, dr=True, seed=1)[3][0], image)


def test_train_reduces_loss_and_exports_working_policy(tmp_path):
    torch.manual_seed(0)
    rows = _rows(tmp_path)
    loader = DataLoader(BCData(rows, dr=False), batch_size=4, shuffle=True)
    encoder, head = Encoder().eval(), PolicyHead()
    history = train(encoder, head, loader, loader, epochs=30, lr=3e-3)
    assert all(math.isfinite(v) for v in history["train"])
    assert history["train"][-1] < history["train"][0]
    path = tmp_path / "bc" / "policy.npz"
    save_npz(path, {**encoder_arrays(encoder), **head_arrays(head)})
    steer = Policy(path).act(np.full(CAMERA_SHAPE, 100, dtype=np.uint8), 0.1)
    assert -1.0 <= steer <= 1.0
