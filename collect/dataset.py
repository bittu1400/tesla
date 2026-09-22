"""Driving data on disk: frame_<n>.png images plus labels.csv, append-only."""
import csv
import re
from pathlib import Path

import cv2
import numpy as np

FIELDS = ("frame", "steer", "prev_steer", "episode", "source")
_FRAME_NAME = re.compile(r"frame_(\d+)\.png$")


def next_frame_index(data_dir) -> int:
    """One past the highest existing frame number (not the file count, so
    deleting bad frames never makes a later run overwrite good ones)."""
    indices = [int(m.group(1)) for p in Path(data_dir).glob("frame_*.png") if (m := _FRAME_NAME.search(p.name))]
    return max(indices, default=-1) + 1


def load_frame(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"unreadable frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class DatasetWriter:
    """Buffers (frame, labels) and writes them every save_every rows, so a
    crash loses at most one batch. Each image is written before its CSV row."""

    def __init__(self, data_dir, save_every: int = 200):
        self.data_dir = Path(data_dir)
        self.save_every = save_every
        self.buffer = []
        self.saved = 0

    def add(self, frame, steer: float, prev_steer: float, episode: str, source: str) -> None:
        self.buffer.append((np.array(frame, dtype=np.uint8), steer, prev_steer, episode, source))
        if len(self.buffer) >= self.save_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        labels = self.data_dir / "labels.csv"
        new_file = not labels.exists()
        start = next_frame_index(self.data_dir)
        with labels.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(FIELDS)
            for i, (frame, steer, prev_steer, episode, source) in enumerate(self.buffer):
                name = f"frame_{start + i:08d}.png"
                if not cv2.imwrite(str(self.data_dir / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"failed to write {self.data_dir / name} (disk full?)")
                writer.writerow([name, f"{steer:.4f}", f"{prev_steer:.4f}", episode, source])
        self.saved += len(self.buffer)
        self.buffer = []
        print(f"saved {self.saved} frames")

    @property
    def total(self) -> int:
        return self.saved + len(self.buffer)


def read_labels(data_dir) -> list[dict]:
    labels = Path(data_dir) / "labels.csv"
    if not labels.exists():
        return []
    with labels.open(newline="") as f:
        return [
            {
                "frame": Path(data_dir) / row["frame"],
                "steer": float(row["steer"]),
                "prev_steer": float(row["prev_steer"]),
                "episode": row["episode"],
                "source": row["source"],
            }
            for row in csv.DictReader(f)
        ]
