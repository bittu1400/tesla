# Lane-Following RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A trained SAC agent that drives the simulated Donkey Car around
a track, staying in its lane, without crashing — evaluated headlessly
with a pass/fail metric.

**Architecture:** gym-donkeycar provides the simulator. A thin Gymnasium
wrapper (`envs/`) crops/normalizes each camera frame and shapes the CTE
signal into a reward. A separately-trained VAE (`vae/`) compresses each
frame to a 32-dim latent vector, which becomes the RL observation.
Stable-Baselines3's SAC trains on that latent space (`rl/`). Scripts
(`scripts/`) drive and evaluate the trained model.

**Tech Stack:** Python 3.11 (venv), gym-donkeycar, gymnasium,
stable-baselines3, torch, numpy, opencv-python, pytest.

**Spec:** [docs/superpowers/specs/2026-09-16-lane-following-rl-design.md](../specs/2026-09-16-lane-following-rl-design.md)

## Global Constraints

- Python 3.11 virtualenv, separate from system Python 3.14 (torch/sb3/gym-donkeycar don't support 3.14).
- Observation fed to SAC = 32-dim VAE latent vector (`float32`).
- Reward per step: `reward = 1.0 - min(abs(cte) / cte_max, 1.0)`, plus `throttle_bonus * speed`.
- Episode ends (treated as `hit`) when the sim reports `hit=True` OR `abs(cte) > cte_max`.
- `models/` (checkpoints) and any collected-frame directories are gitignored.
- Training/eval scripts must catch sim disconnects and retry rather than crash.
- Obstacle avoidance, extra sensors, and real-car deployment are OUT OF SCOPE for this plan (spec's Future Work) — do not build them.

---

## File Structure

```
Tesla/
  requirements.txt
  .gitignore
  README.md
  envs/
    __init__.py
    preprocessing.py      # crop_resize_normalize(frame) -> np.ndarray
    reward.py              # compute_reward(cte, hit, speed, cte_max) -> (reward, done)
    donkey_env.py           # DonkeyLaneEnv(gymnasium.Env)
  vae/
    __init__.py
    model.py                # ConvVAE(nn.Module), encode/decode
    frame_store.py           # save_frames(frames, out_dir) -> int
    collect_frames.py         # CLI: drive sim, dump frames via frame_store
    train_vae.py               # train(model, dataloader, epochs, device) -> list[float]; CLI entrypoint
  rl/
    __init__.py
    config.py                 # SAC_CONFIG dict
    train_sac.py                # build_env(), build_model(), CLI entrypoint
  scripts/
    check_sim.py                # manual connectivity smoke test (like test_carla.py)
    eval.py                      # run_eval(model, env, n_episodes) -> dict; CLI entrypoint
    drive.py                      # CLI: load model, run in sim for viewing
  tests/
    test_preprocessing.py
    test_reward.py
    test_donkey_env.py
    test_frame_store.py
    test_vae_model.py
    test_train_vae.py
    test_train_sac.py
    test_eval.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `envs/__init__.py`, `vae/__init__.py`, `rl/__init__.py` (empty)
- Create: `scripts/` and `tests/` directories (no `__init__.py` needed)

**Interfaces:**
- Produces: nothing importable yet — this is the skeleton later tasks fill in.

- [ ] **Step 1: Write `requirements.txt`**

```
gym-donkeycar
gymnasium
stable-baselines3
torch
numpy
opencv-python
pytest
```

- [ ] **Step 2: Write `.gitignore`**

```
models/
*.pyc
__pycache__/
.venv/
venv/
collected_frames/
*.egg-info/
```

- [ ] **Step 3: Write `README.md`**

```markdown
# Tesla — Lane-Following RL for Donkey Car

Simulated lane-following RL agent (gym-donkeycar + SAC), intended to
later transfer to a real Donkey Car. See
`docs/superpowers/specs/2026-09-16-lane-following-rl-design.md` for
the full design.

## Setup

1. Create a Python 3.11 venv: `python3.11 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Download the Donkey Car simulator binary (Unity build) per
   gym-donkeycar's install docs: https://github.com/tawnkramer/gym-donkeycar/releases
   Extract it somewhere and note the executable path.
4. Run `python scripts/check_sim.py --exe-path /path/to/sim/binary` to confirm
   the simulator launches and connects (starts the sim if not already running).

## Training

1. `python -m vae.collect_frames --exe-path ... --out-dir collected_frames --n-frames 8000`
2. `python -m vae.train_vae --frames-dir collected_frames --out models/vae.pth`
3. `python -m rl.train_sac --exe-path ... --vae-path models/vae.pth --out models/sac`
4. `python -m scripts.eval --exe-path ... --vae-path models/vae.pth --model-path models/sac.zip`
```

- [ ] **Step 4: Create empty package files and directories**

```bash
mkdir -p envs vae rl scripts tests
touch envs/__init__.py vae/__init__.py rl/__init__.py
```

- [ ] **Step 5: Create and activate the venv, install deps**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: install completes with no errors. If `python3.11` isn't
found, install it first (e.g. `pyenv install 3.11.9` or system package
manager) — this is an environment prerequisite, not a plan bug.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore README.md envs vae rl scripts tests
git commit -m "chore: project scaffolding for lane-following RL"
```

---

### Task 2: Simulator connectivity smoke test

**Files:**
- Create: `scripts/check_sim.py`

**Interfaces:**
- Produces: nothing importable — standalone manual-verification script.

- [ ] **Step 1: Write `scripts/check_sim.py`**

```python
"""Manual smoke test: confirms the Donkey Car simulator launches and
gym-donkeycar can connect to it. Run this once after installing the
sim binary, before writing any training code that depends on it."""
import argparse

import gym_donkeycar  # noqa: F401  (registers the donkey-* gym envs)
import gymnasium as gym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-path", required=True, help="Path to the sim binary")
    parser.add_argument("--env-name", default="donkey-generated-track-v0")
    args = parser.parse_args()

    env = gym.make(
        args.env_name,
        conf={"exe_path": args.exe_path, "port": 9091},
    )
    obs, info = env.reset()
    print("Connected to Donkey Car sim!")
    print("Observation shape:", obs.shape)
    print("Info keys:", list(info.keys()))
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it manually against the downloaded sim binary**

Run: `python scripts/check_sim.py --exe-path /path/to/sim/binary`
Expected: prints "Connected to Donkey Car sim!", an observation shape
like `(120, 160, 3)`, and info keys including `cte` and `hit`. If the
keys differ, note the actual names — later tasks assume `cte`/`hit`
and must be adjusted if the installed gym-donkeycar version differs.

- [ ] **Step 3: Commit**

```bash
git add scripts/check_sim.py
git commit -m "feat: add simulator connectivity smoke test"
```

---

### Task 3: Frame preprocessing (pure function)

**Files:**
- Create: `envs/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `crop_resize_normalize(frame: np.ndarray, size: tuple[int, int] = (80, 80)) -> np.ndarray`
  — input HxWx3 `uint8` array, output `size[0] x size[1] x 3` `float32` array with values in `[0, 1]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessing.py
import numpy as np

from envs.preprocessing import crop_resize_normalize


def test_output_shape_and_dtype():
    frame = np.random.randint(0, 256, size=(120, 160, 3), dtype=np.uint8)
    out = crop_resize_normalize(frame, size=(80, 80))
    assert out.shape == (80, 80, 3)
    assert out.dtype == np.float32


def test_values_normalized_to_unit_range():
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)
    out = crop_resize_normalize(frame, size=(80, 80))
    assert out.max() <= 1.0
    assert out.min() >= 0.0
    # a fully-white input should normalize to ~1.0 everywhere
    assert np.allclose(out, 1.0, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` (function doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# envs/preprocessing.py
import cv2
import numpy as np


def crop_resize_normalize(frame: np.ndarray, size: tuple[int, int] = (80, 80)) -> np.ndarray:
    """Resize a raw camera frame and normalize to [0, 1] float32.

    No cropping is applied by default (sim frames are already mostly
    track); pass a pre-cropped frame in if the real camera needs it.
    """
    resized = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32)) / 255.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocessing.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add envs/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add frame preprocessing (crop/resize/normalize)"
```

---

### Task 4: Reward shaping (pure function)

**Files:**
- Create: `envs/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Produces: `compute_reward(cte: float, hit: bool, speed: float, cte_max: float, throttle_bonus: float = 0.1) -> tuple[float, bool]`
  — returns `(reward, done)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py
from envs.reward import compute_reward


def test_centered_lane_gives_near_max_reward():
    reward, done = compute_reward(cte=0.0, hit=False, speed=1.0, cte_max=2.0)
    assert not done
    assert reward > 1.0  # 1.0 base + throttle bonus


def test_edge_of_lane_gives_near_zero_base_reward():
    reward, done = compute_reward(cte=2.0, hit=False, speed=0.0, cte_max=2.0)
    assert not done
    assert abs(reward) < 1e-6


def test_hit_ends_episode_with_negative_reward():
    reward, done = compute_reward(cte=0.0, hit=True, speed=1.0, cte_max=2.0)
    assert done
    assert reward < 0


def test_cte_beyond_max_treated_as_hit():
    reward, done = compute_reward(cte=3.0, hit=False, speed=1.0, cte_max=2.0)
    assert done
    assert reward < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# envs/reward.py
HIT_PENALTY = -5.0


def compute_reward(
    cte: float,
    hit: bool,
    speed: float,
    cte_max: float,
    throttle_bonus: float = 0.1,
) -> tuple[float, bool]:
    """Reward shaping for lane-following.

    1.0 at lane center, decaying to 0.0 at cte_max, plus a small
    speed-proportional bonus so the agent doesn't learn to sit still.
    Collision or leaving the lane ends the episode with a fixed penalty.
    """
    off_lane = abs(cte) > cte_max
    if hit or off_lane:
        return HIT_PENALTY, True

    base = 1.0 - min(abs(cte) / cte_max, 1.0)
    reward = base + throttle_bonus * speed
    return reward, False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add envs/reward.py tests/test_reward.py
git commit -m "feat: add lane-following reward shaping"
```

---

### Task 5: DonkeyLaneEnv wrapper (raw-frame observation)

**Files:**
- Create: `envs/donkey_env.py`
- Test: `tests/test_donkey_env.py`

**Interfaces:**
- Consumes: `crop_resize_normalize` from `envs/preprocessing.py`, `compute_reward` from `envs/reward.py`.
- Produces: `DonkeyLaneEnv(gymnasium.Env)` with constructor
  `DonkeyLaneEnv(exe_path: str, env_name: str = "donkey-generated-track-v0", cte_max: float = 2.0, frame_size: tuple[int,int] = (80,80), port: int = 9091)`,
  standard `reset() -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)`.
  `obs` is the *raw preprocessed frame* (`frame_size[0] x frame_size[1] x 3` float32) at this stage —
  Task 9 swaps this for the VAE latent without changing the class's public interface.
  Reconnect-on-disconnect: `reset()` and `step()` catch connection errors from the
  underlying env and retry the sim connection once before raising.

This task wires to the real `gym_donkeycar` package, so the unit test
mocks the underlying env (no simulator process required to run
`pytest`); Task 2's `check_sim.py` already confirmed the real
connection works.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_donkey_env.py
import numpy as np
import pytest

from envs.donkey_env import DonkeyLaneEnv


class FakeUnderlyingEnv:
    """Stands in for the gym_donkeycar env `gym.make()` would return."""

    def __init__(self, frame_shape=(120, 160, 3)):
        self.frame_shape = frame_shape
        self.closed = False

    def reset(self, **kwargs):
        obs = np.zeros(self.frame_shape, dtype=np.uint8)
        return obs, {"cte": 0.0, "hit": False, "speed": 0.0}

    def step(self, action):
        obs = np.zeros(self.frame_shape, dtype=np.uint8)
        info = {"cte": 0.5, "hit": False, "speed": 2.0}
        return obs, 0.0, False, False, info

    def close(self):
        self.closed = True


@pytest.fixture
def env(monkeypatch):
    fake = FakeUnderlyingEnv()
    monkeypatch.setattr(
        "envs.donkey_env._make_underlying_env", lambda *a, **kw: fake
    )
    return DonkeyLaneEnv(exe_path="/fake/path", cte_max=2.0, frame_size=(80, 80))


def test_reset_returns_preprocessed_observation(env):
    obs, info = env.reset()
    assert obs.shape == (80, 80, 3)
    assert obs.dtype == np.float32


def test_step_returns_shaped_reward(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert obs.shape == (80, 80, 3)
    # cte=0.5, cte_max=2.0, speed=2.0 -> matches compute_reward directly
    from envs.reward import compute_reward
    expected_reward, expected_done = compute_reward(0.5, False, 2.0, 2.0)
    assert reward == pytest.approx(expected_reward)
    assert terminated == expected_done


def test_step_on_hit_terminates(env, monkeypatch):
    env.reset()

    def hit_step(action):
        obs = np.zeros((120, 160, 3), dtype=np.uint8)
        return obs, 0.0, False, False, {"cte": 0.0, "hit": True, "speed": 0.0}

    monkeypatch.setattr(env._underlying, "step", hit_step)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert terminated is True
    assert reward < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_donkey_env.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# envs/donkey_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.preprocessing import crop_resize_normalize
from envs.reward import compute_reward


def _make_underlying_env(exe_path: str, env_name: str, port: int):
    """Thin factory around gym.make so tests can monkeypatch it."""
    import gym_donkeycar  # noqa: F401  (registers donkey-* envs)

    return gym.make(env_name, conf={"exe_path": exe_path, "port": port}).unwrapped


class DonkeyLaneEnv(gym.Env):
    """Gymnasium wrapper around gym-donkeycar for lane-following.

    Observation is the preprocessed camera frame (VAE encoding is
    applied by a later wrapper, not here — see vae integration).
    """

    def __init__(
        self,
        exe_path: str,
        env_name: str = "donkey-generated-track-v0",
        cte_max: float = 2.0,
        frame_size: tuple[int, int] = (80, 80),
        port: int = 9091,
    ):
        super().__init__()
        self.exe_path = exe_path
        self.env_name = env_name
        self.cte_max = cte_max
        self.frame_size = frame_size
        self.port = port

        self._underlying = _make_underlying_env(exe_path, env_name, port)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(*frame_size, 3), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

    def _reconnect(self):
        self._underlying = _make_underlying_env(self.exe_path, self.env_name, self.port)

    def reset(self, **kwargs):
        try:
            raw_obs, info = self._underlying.reset(**kwargs)
        except (ConnectionError, BrokenPipeError):
            self._reconnect()
            raw_obs, info = self._underlying.reset(**kwargs)
        obs = crop_resize_normalize(raw_obs, self.frame_size)
        return obs, info

    def step(self, action):
        try:
            raw_obs, _sim_reward, _term, _trunc, info = self._underlying.step(action)
        except (ConnectionError, BrokenPipeError):
            self._reconnect()
            raw_obs, _sim_reward, _term, _trunc, info = self._underlying.step(action)

        obs = crop_resize_normalize(raw_obs, self.frame_size)
        reward, terminated = compute_reward(
            cte=info.get("cte", 0.0),
            hit=info.get("hit", False),
            speed=info.get("speed", 0.0),
            cte_max=self.cte_max,
        )
        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self):
        self._underlying.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_donkey_env.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add envs/donkey_env.py tests/test_donkey_env.py
git commit -m "feat: add DonkeyLaneEnv gymnasium wrapper"
```

---

### Task 6: VAE model

**Files:**
- Create: `vae/model.py`
- Test: `tests/test_vae_model.py`

**Interfaces:**
- Produces: `ConvVAE(nn.Module)` with constructor `ConvVAE(latent_dim: int = 32, input_size: tuple[int,int] = (80, 80))`,
  methods `encode(x: torch.Tensor) -> torch.Tensor` (returns `(batch, latent_dim)`),
  `decode(z: torch.Tensor) -> torch.Tensor` (returns `(batch, 3, input_size[0], input_size[1])`),
  `forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` (returns `(reconstruction, mu, logvar)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vae_model.py
import torch

from vae.model import ConvVAE


def test_encode_shape():
    model = ConvVAE(latent_dim=32, input_size=(80, 80))
    x = torch.rand(4, 3, 80, 80)
    z = model.encode(x)
    assert z.shape == (4, 32)


def test_decode_shape():
    model = ConvVAE(latent_dim=32, input_size=(80, 80))
    z = torch.rand(4, 32)
    out = model.decode(z)
    assert out.shape == (4, 3, 80, 80)


def test_forward_roundtrip_shapes():
    model = ConvVAE(latent_dim=32, input_size=(80, 80))
    x = torch.rand(4, 3, 80, 80)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, 32)
    assert logvar.shape == (4, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vae_model.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# vae/model.py
import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    """Small convolutional VAE for compressing 80x80 camera frames."""

    def __init__(self, latent_dim: int = 32, input_size: tuple[int, int] = (80, 80)):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 80 -> 40
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 40 -> 20
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 20 -> 10
            nn.ReLU(),
        )
        self.flat_size = 128 * 10 * 10
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flat_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 10 -> 20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 20 -> 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 40 -> 80
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the mean latent vector (used at inference time)."""
        h = self.encoder_conv(x).flatten(1)
        return self.fc_mu(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, 128, 10, 10)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        h = self.encoder_conv(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder_conv(self.fc_decode(z).view(-1, 128, 10, 10))
        return recon, mu, logvar
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vae_model.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add vae/model.py tests/test_vae_model.py
git commit -m "feat: add ConvVAE model"
```

---

### Task 7: Frame storage + collection script

**Files:**
- Create: `vae/frame_store.py`
- Create: `vae/collect_frames.py`
- Test: `tests/test_frame_store.py`

**Interfaces:**
- Produces: `save_frames(frames: list[np.ndarray], out_dir: Path) -> int` (returns count of frames written, as `frame_00000.png`, `frame_00001.png`, ...).
- `collect_frames.py` is a CLI script (no importable interface consumed elsewhere); it drives `DonkeyLaneEnv` with random actions and calls `save_frames` periodically. Manual verification only (needs the real sim).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame_store.py
import numpy as np

from vae.frame_store import save_frames


def test_save_frames_writes_expected_count(tmp_path):
    frames = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(5)]
    count = save_frames(frames, tmp_path)
    assert count == 5
    assert len(list(tmp_path.glob("*.png"))) == 5


def test_save_frames_appends_without_overwriting(tmp_path):
    frames_a = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(3)]
    frames_b = [np.ones((80, 80, 3), dtype=np.uint8) for _ in range(2)]
    save_frames(frames_a, tmp_path)
    save_frames(frames_b, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_store.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# vae/frame_store.py
from pathlib import Path

import cv2
import numpy as np


def save_frames(frames: list[np.ndarray], out_dir: Path) -> int:
    """Write frames as sequential PNGs, continuing numbering from any
    frames already in out_dir so repeated collection runs append."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.png"))
    start = len(existing)

    for i, frame in enumerate(frames):
        path = out_dir / f"frame_{start + i:05d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return len(frames)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_store.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write `vae/collect_frames.py`**

```python
# vae/collect_frames.py
"""Drive the sim with random actions, periodically dumping camera
frames to disk for later VAE training. Manual step — run this once
against the real simulator, then train_vae.py on the output."""
import argparse
from pathlib import Path

import numpy as np

from envs.donkey_env import DonkeyLaneEnv
from vae.frame_store import save_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-frames", type=int, default=8000)
    parser.add_argument("--save-every", type=int, default=200)
    args = parser.parse_args()

    env = DonkeyLaneEnv(exe_path=args.exe_path)
    obs, _ = env.reset()
    buffer = []
    collected = 0

    while collected < args.n_frames:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        buffer.append((obs * 255).astype(np.uint8))
        collected += 1

        if len(buffer) >= args.save_every:
            save_frames(buffer, Path(args.out_dir))
            buffer = []
            print(f"Collected {collected}/{args.n_frames} frames")

        if terminated or truncated:
            obs, _ = env.reset()

    if buffer:
        save_frames(buffer, Path(args.out_dir))
    env.close()
    print(f"Done. {collected} frames saved to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add vae/frame_store.py vae/collect_frames.py tests/test_frame_store.py
git commit -m "feat: add frame storage and collection script"
```

---

### Task 8: VAE training

**Files:**
- Create: `vae/train_vae.py`
- Test: `tests/test_train_vae.py`

**Interfaces:**
- Consumes: `ConvVAE` from `vae/model.py`.
- Produces: `vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor`,
  `train(model: ConvVAE, dataloader: DataLoader, epochs: int, device: str = "cpu") -> list[float]` (per-epoch average loss).
  CLI entrypoint loads images from a directory via `FrameDataset`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_vae.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from vae.model import ConvVAE
from vae.train_vae import train, vae_loss


def test_vae_loss_is_finite_scalar():
    model = ConvVAE(latent_dim=32, input_size=(80, 80))
    x = torch.rand(4, 3, 80, 80)
    recon, mu, logvar = model(x)
    loss = vae_loss(recon, x, mu, logvar)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_train_returns_one_loss_per_epoch():
    model = ConvVAE(latent_dim=32, input_size=(80, 80))
    data = torch.rand(8, 3, 80, 80)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=4)

    losses = train(model, loader, epochs=2, device="cpu")

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_vae.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# vae/train_vae.py
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from vae.model import ConvVAE


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld


def train(model: ConvVAE, dataloader: DataLoader, epochs: int, device: str = "cpu") -> list[float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for _epoch in range(epochs):
        total = 0.0
        n_batches = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        losses.append(total / n_batches)

    return losses


class FrameDataset(Dataset):
    """Loads PNG frames from a directory as (3, H, W) float32 tensors in [0, 1]."""

    def __init__(self, frames_dir: str, input_size: tuple[int, int] = (80, 80)):
        self.paths = sorted(Path(frames_dir).glob("*.png"))
        self.input_size = input_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FrameDataset(args.frames_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ConvVAE(latent_dim=args.latent_dim)
    losses = train(model, loader, epochs=args.epochs, device=device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Trained {args.epochs} epochs, final loss {losses[-1]:.4f}, saved to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_vae.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add vae/train_vae.py tests/test_train_vae.py
git commit -m "feat: add VAE training loop and CLI"
```

---

### Task 9: Wire VAE into DonkeyLaneEnv

**Files:**
- Modify: `envs/donkey_env.py`
- Modify: `tests/test_donkey_env.py`

**Interfaces:**
- Consumes: `ConvVAE.encode` from `vae/model.py`.
- Produces: `DonkeyLaneEnv.__init__` gains `vae_path: str | None = None` param. When set, `observation_space`
  becomes `Box(low=-inf, high=inf, shape=(latent_dim,), dtype=np.float32)` and `reset`/`step` return the VAE
  latent instead of the raw preprocessed frame. When `vae_path=None`, behavior is unchanged from Task 5
  (needed for VAE-free tests and for `collect_frames.py`, which must not depend on a trained VAE).

- [ ] **Step 1: Extend the test file**

Add to `tests/test_donkey_env.py`:

```python
def test_reset_returns_latent_when_vae_path_given(monkeypatch, tmp_path):
    import torch
    from vae.model import ConvVAE

    vae_path = tmp_path / "vae.pth"
    torch.save(ConvVAE(latent_dim=32).state_dict(), vae_path)

    fake = FakeUnderlyingEnv()
    monkeypatch.setattr(
        "envs.donkey_env._make_underlying_env", lambda *a, **kw: fake
    )
    env = DonkeyLaneEnv(
        exe_path="/fake/path", cte_max=2.0, frame_size=(80, 80), vae_path=str(vae_path)
    )
    obs, info = env.reset()
    assert obs.shape == (32,)
    assert obs.dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_donkey_env.py -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'vae_path'`)

- [ ] **Step 3: Modify `envs/donkey_env.py`**

```python
# Add near the top imports:
import torch

from vae.model import ConvVAE

# Replace __init__ and add an encode helper:
    def __init__(
        self,
        exe_path: str,
        env_name: str = "donkey-generated-track-v0",
        cte_max: float = 2.0,
        frame_size: tuple[int, int] = (80, 80),
        port: int = 9091,
        vae_path: str | None = None,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.exe_path = exe_path
        self.env_name = env_name
        self.cte_max = cte_max
        self.frame_size = frame_size
        self.port = port
        self.vae_path = vae_path
        self.latent_dim = latent_dim

        self._underlying = _make_underlying_env(exe_path, env_name, port)

        self._vae = None
        if vae_path is not None:
            self._vae = ConvVAE(latent_dim=latent_dim, input_size=frame_size)
            self._vae.load_state_dict(torch.load(vae_path, map_location="cpu"))
            self._vae.eval()

        if self._vae is not None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(*frame_size, 3), dtype=np.float32
            )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

    def _encode(self, frame: np.ndarray) -> np.ndarray:
        if self._vae is None:
            return frame
        with torch.no_grad():
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
            latent = self._vae.encode(tensor)
        return latent.squeeze(0).numpy().astype(np.float32)
```

Then update `reset` and `step` to call `self._encode(obs)` on the
preprocessed frame before returning it (replace the line
`obs = crop_resize_normalize(raw_obs, self.frame_size)` with that same
line followed by `obs = self._encode(obs)` in both methods).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_donkey_env.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add envs/donkey_env.py tests/test_donkey_env.py
git commit -m "feat: wire VAE latent encoding into DonkeyLaneEnv"
```

---

### Task 10: SAC training

**Files:**
- Create: `rl/config.py`
- Create: `rl/train_sac.py`
- Test: `tests/test_train_sac.py`

**Interfaces:**
- Consumes: `DonkeyLaneEnv` from `envs/donkey_env.py`.
- Produces: `SAC_CONFIG: dict` in `rl/config.py`; `build_model(env, config: dict) -> stable_baselines3.SAC` in `rl/train_sac.py`.

- [ ] **Step 1: Write `rl/config.py`**

```python
# rl/config.py
SAC_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 50_000,
    "batch_size": 256,
    "train_freq": 1,
    "gradient_steps": 1,
    "learning_starts": 1000,
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_train_sac.py
import numpy as np
import pytest
from stable_baselines3 import SAC

from rl.config import SAC_CONFIG
from rl.train_sac import build_model


class FakeLatentEnv:
    """Minimal gymnasium-like env for a smoke-testing SAC's wiring,
    without needing the real simulator or a trained VAE."""

    def __init__(self):
        import gymnasium as gym
        from gymnasium import spaces

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(32,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._steps += 1
        terminated = self._steps >= 5
        return self.observation_space.sample(), 0.1, terminated, False, {}


def test_build_model_matches_env_spaces():
    env = FakeLatentEnv()
    model = build_model(env, SAC_CONFIG)
    assert model.observation_space.shape == (32,)
    assert model.action_space.shape == (2,)


def test_short_learn_run_does_not_crash():
    env = FakeLatentEnv()
    model = build_model(env, SAC_CONFIG)
    model.learn(total_timesteps=20)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_train_sac.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Write `rl/train_sac.py`**

```python
# rl/train_sac.py
import argparse

from stable_baselines3 import SAC

from envs.donkey_env import DonkeyLaneEnv
from rl.config import SAC_CONFIG


def build_env(exe_path: str, vae_path: str) -> DonkeyLaneEnv:
    return DonkeyLaneEnv(exe_path=exe_path, vae_path=vae_path)


def build_model(env, config: dict) -> SAC:
    return SAC(env=env, verbose=1, **config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    args = parser.parse_args()

    env = build_env(args.exe_path, args.vae_path)
    model = build_model(env, SAC_CONFIG)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.out)
    env.close()
    print(f"Saved trained model to {args.out}.zip")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train_sac.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add rl/config.py rl/train_sac.py tests/test_train_sac.py
git commit -m "feat: add SAC training script"
```

---

### Task 11: Evaluation script

**Files:**
- Create: `scripts/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces: `run_eval(model, env, n_episodes: int) -> dict` — returns
  `{"success_rate": float, "avg_abs_cte": float, "max_abs_cte": float}`,
  where `success_rate` is the fraction of episodes that ended via
  `truncated` (max steps reached) rather than `terminated` (a `hit`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval.py
import numpy as np

from scripts.eval import run_eval


class FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([0.0, 0.5]), None


class FakeEvalEnv:
    """Always survives 3 steps then truncates, with a known cte in info."""

    def __init__(self):
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return np.zeros(32, dtype=np.float32), {"cte": 0.0}

    def step(self, action):
        self._steps += 1
        truncated = self._steps >= 3
        info = {"cte": 0.5}
        return np.zeros(32, dtype=np.float32), 1.0, False, truncated, info


def test_run_eval_reports_success_rate_and_cte():
    model = FakeModel()
    env = FakeEvalEnv()
    result = run_eval(model, env, n_episodes=2)
    assert result["success_rate"] == 1.0
    assert result["avg_abs_cte"] == 0.5
    assert result["max_abs_cte"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `scripts/eval.py`**

```python
# scripts/eval.py
import argparse

from stable_baselines3 import SAC

from envs.donkey_env import DonkeyLaneEnv


def run_eval(model, env, n_episodes: int) -> dict:
    successes = 0
    abs_ctes = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            abs_ctes.append(abs(info.get("cte", 0.0)))
        if truncated and not terminated:
            successes += 1

    return {
        "success_rate": successes / n_episodes,
        "avg_abs_cte": sum(abs_ctes) / len(abs_ctes) if abs_ctes else 0.0,
        "max_abs_cte": max(abs_ctes) if abs_ctes else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    env = DonkeyLaneEnv(exe_path=args.exe_path, vae_path=args.vae_path)
    model = SAC.load(args.model_path)
    result = run_eval(model, env, args.n_episodes)
    env.close()

    print(f"Success rate: {result['success_rate']:.1%}")
    print(f"Avg |CTE|: {result['avg_abs_cte']:.3f}")
    print(f"Max |CTE|: {result['max_abs_cte']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/eval.py tests/test_eval.py
git commit -m "feat: add headless evaluation script"
```

---

### Task 12: Drive/watch script

**Files:**
- Create: `scripts/drive.py`

**Interfaces:**
- Consumes: `DonkeyLaneEnv` from `envs/donkey_env.py`. CLI-only, no importable interface consumed by later tasks — manual verification (watching the sim window) is the deliverable check.

- [ ] **Step 1: Write `scripts/drive.py`**

```python
# scripts/drive.py
"""Run a trained model in the simulator so you can watch it drive.
Manual verification: the sim window should show the car following the
lane. No automated test — this is a viewing tool, not new logic
(reuses DonkeyLaneEnv and SAC.load, already tested elsewhere)."""
import argparse

from stable_baselines3 import SAC

from envs.donkey_env import DonkeyLaneEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = DonkeyLaneEnv(exe_path=args.exe_path, vae_path=args.vae_path)
    model = SAC.load(args.model_path)

    for ep in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep}: reward={total_reward:.2f}, ended_by={'hit' if terminated else 'timeout'}")

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it manually against a trained checkpoint**

Run: `python scripts/drive.py --exe-path ... --vae-path models/vae.pth --model-path models/sac.zip`
Expected: sim window opens, car drives, per-episode summary lines print. Confirms the full pipeline end to end.

- [ ] **Step 3: Commit**

```bash
git add scripts/drive.py
git commit -m "feat: add drive/watch script for trained models"
```

---

## Self-Review Notes

- **Spec coverage:** project structure (Task 1), env/tooling (Task 1),
  reward function (Task 4), VAE (Tasks 6-8), VAE wired into env (Task 9),
  RL training (Task 10), evaluation criteria (Task 11), reconnect-on-disconnect
  (Task 5), testing approach (sanity tests throughout, matches spec's
  "sanity script" + "eval.py is the check" guidance), future work
  explicitly excluded (Global Constraints) — all spec sections covered.
- **Placeholder scan:** no TBD/TODO; every step has concrete code.
- **Type consistency:** `DonkeyLaneEnv` constructor signature is
  introduced in Task 5 and extended (not renamed) in Task 9;
  `compute_reward`'s signature is identical everywhere it's referenced
  (Tasks 4, 5); `ConvVAE.encode`/`decode` signatures from Task 6 are
  used as-is in Tasks 8 and 9; `run_eval` and `build_model` signatures
  match their test usage.
