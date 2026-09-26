# Sim-to-Real Robustness Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Code for a 2×2 benchmark (SAC vs BC × clean vs domain-randomized) that trains four steering policies in the Donkey simulator, tests them under six noise conditions in the simulator and on the real donkeycar, and produces the statistics and figures for the paper.

**Architecture:** `car/policy.py` is numpy-only inference (preprocess → VAE encoder → MLP head) shared by the simulator, the benchmark and the Raspberry Pi. `envs/` wraps gym-donkeycar as a steering-only env with a fixed throttle, plus wrappers for latent observations, actuator lag and image augmentation. `collect/` records labeled driving data, `learn/` trains VAEs, BC heads and SAC (all exported to the same `policy.npz` format), `bench/` runs the simulator benchmark, schedules and analyses real trials, and `car/` holds the donkeycar part and the trial runner used on the Pi.

**Tech Stack:** Python 3.12 (uv venv), gym-donkeycar (git-pinned, gymnasium API), gymnasium 1.3, stable-baselines3 2.9, torch 2.14, numpy 2, opencv-python-headless, pygame, scipy, matplotlib, pytest. On the Pi: donkeycar 5.x `complete` template, numpy, Pillow (both come with donkeycar).

**Spec:** [docs/superpowers/specs/2026-09-16-sim2real-robustness-benchmark-design.md](../specs/2026-09-16-sim2real-robustness-benchmark-design.md)

## Status (2026-09-26)

**All 17 tasks are done.** Branch `sim2real-benchmark` was fast-forwarded into `master` and deleted. Post-plan commits on `master`:
- `1b6f656`..`a5ebc1d`: fixes from the final review.
- `28dec99`..`e4d1027`: fixes found while calibrating on the real sim.
- `cbeb974`, `8e57d6d`: the speed work below. `c0a6cb1`: docs. Pushed to `origin/master` on 2026-09-26.
- `e8c9703` (2026-09-26): PD gain flags for collection and multi-dir training data. It and the docs commit after it are **not pushed**.

The suite has 134 tests. The SDD ledger (`.superpowers/sdd/...`) was deleted; git history is the record now.

The manual steps ran on the real sim on 2026-09-23:
- Task 6 Steps 2–4: calibration.
- Task 9 Step 7: both collection modes start and record. The user has not yet driven with the arrow keys.
- Task 17: smoke run.

**The code has diverged from the listings below.** The repository is the source of truth. Changes after the plan:
- `envs/donkey_env.py`: the `cte_offset` flag, env-side off-lane termination, and `SETTLE_STEPS`.
- `collect/drivers.py`: PD control and 3-step swerves.
- `bench/sim_bench.py`: `--workers N` (ports skip 9092; every sim also opens 0.0.0.0:9092), and `run_bench(..., shard=(i, n))`.
- `envs/sim_process.py`: the log file is `unitylog_<port>.txt`.
- `collect/collect.py`: `--gain` and `--d-gain` for the scripted driver (defaults 0.3 and 4.0, as before).
- `learn/train_vae.py`, `learn/train_bc.py`: `--data-dir` takes one or more dirs and concatenates them.

What was run after the plan (data, training, gate, benchmark, both track extensions, the camera FOV study), every deviation with its reason, the results, and the open decisions: design spec, section **"As run"**. Commands: `README.md`. Per-step runbook: `todo.md`, which is gitignored and on the laptop only.

Rulings still standing:
- **Sim `offset_start`:** at handover the policy's `prev_steer` input is `±0.6`, the command executed during the forced drift. It was kept, because everywhere else `prev_steer` means "previous executed command". To revisit, change one line in `bench/sim_bench.py::run_episode`.
- **`.gitignore`:** also ignores `Donkey Car Simulation Research Topics.pdf`.

## Global Constraints

- Python 3.12 venv created with `uv` at `.venv/`; never install into system Python 3.14.
- gym-donkeycar from git commit `a1f4ca6961e17a6929c1f2e883358a1edafb479d` (gymnasium API: `reset() -> (obs, info)`, `step() -> (obs, reward, terminated, truncated, info)`).
- gym-donkeycar facts the code relies on (checked in its source at that commit):
  - `info["hit"]` is a **string**: `"none"` when nothing was hit. Never test it for truthiness.
  - `info` keys used: `cte`, `hit`, `forward_vel`, `lap_count`, `last_lap_time`, `pos`.
  - The sim's own game-over fires on `|cte| > max_cte` (ignoring spikes above `2 * max_cte` just after reset), a hit, a missed checkpoint or disqualification. It never sets `truncated`.
  - `observe()` busy-waits forever if the sim stops sending frames.
  - `conf["cam_config"]` accepts `img_w`, `img_h`, `fov` (and more).
- Camera frames are RGB uint8 `(120, 160, 3)` in sim and on the car.
- Preprocessing exists only in `car/policy.py::preprocess` (crop top 40 rows, average column pairs → `3×80×80` float32 in [0, 1]). Nothing else crops or resizes.
- Policy action = steering only, in [-1, 1]. Throttle is a fixed constant (`DonkeyLaneEnv.throttle` in sim, `car/settings.json` on the car).
- Policy observation = `[32-dim encoder latent, previous steering command]` (33 floats).
- Head = `Linear(33,64)-ReLU-Linear(64,64)-ReLU-Linear(64,1)`, mean clamped to [-2, 2], then tanh. Weight keys in `.npz`: `encoder.conv.{0,2,4}.{weight,bias}`, `encoder.fc_mu.{weight,bias}`, `head.latent_pi.{0,2}.{weight,bias}`, `head.mu.{weight,bias}`.
- Every trained model ends as `models/<name>/policy.npz`, names: `sac_clean`, `sac_dr`, `bc_clean`, `bc_dr`.
- `car/` imports only the standard library, numpy and (lazily) PIL. Never import torch, cv2 or anything from `envs/`, `learn/`, `bench/` in `car/`.
- Condition names, identical in sim and real: `nominal`, `low_light`, `shadows`, `distractors`, `low_battery`, `offset_start`.
- Run every command from the repo root with `.venv` active, as `python -m package.module`.
- `data/`, `models/`, `results/`, `demo/`, `unitylog*.txt` and `todo.md` are gitignored. Never commit them.
- Commits: plain messages, no co-author or tool attribution lines. Code commits never include `docs/`; doc changes go in their own `docs:` commit, and only when the user asks.
- Unit tests never start the real simulator or need the Pi. Steps marked **manual** need a person.

---

## File Structure

```
Tesla/
  requirements.txt  pytest.ini  README.md  .gitignore
  car/
    __init__.py
    policy.py          # CAMERA_SHAPE, CROP_TOP, CLIP_MEAN, preprocess, conv2d, Encoder, Head, Policy, load_arrays
    pilot_part.py      # RobustPilot (donkeycar part)
    trial.py           # one real trial on the Pi
    settings.json      # fixed car throttle, donkeycar dir, image logging rate
    schedule.csv       # generated by bench.schedule (Task 15)
  envs/
    __init__.py
    reward.py          # CRASH_PENALTY, is_hit, compute_reward
    sim_process.py     # SimDisconnectedError, call_with_timeout, launch_sim, kill_sim
    donkey_env.py      # DonkeyLaneEnv (steering-only, raw frames)
    cli.py             # add_env_args, env_kwargs_from_args
    augment.py         # DomainRandomizer, LowLight, MovingShadow, Distractors, make_augmenter
    wrappers.py        # ActuatorLag, LatentEnv
  collect/
    __init__.py
    drivers.py         # ScriptedDriver, keys_to_steer
    dataset.py         # DatasetWriter, read_labels, load_frame, next_frame_index
    collect.py         # run_scripted, run_manual, CLI
  learn/
    __init__.py
    nets.py            # Encoder, ConvVAE, PolicyHead, frames_to_tensor, save/load_vae, encoder_arrays, head_arrays, save_npz
    train_vae.py       # DriveFrames, vae_loss, run_epoch, train, save_reconstructions, CLI
    train_bc.py        # BCData, split_by_episode, run_epoch, train, CLI
    sac_config.py      # SAC_CONFIG
    train_sac.py       # latest_checkpoint, train, export_sac_policy, CLI
  bench/
    __init__.py
    conditions.py      # Condition, CONDITIONS, BY_NAME, MODELS
    metrics.py         # steering_jerk, oscillation_hz, wilson_interval, parse_trial_log, trial_outcome
    sim_bench.py       # run_episode, plan_runs, run_bench, CLI
    schedule.py        # make_schedule, write_schedule, CLI
    report.py          # load_sim, load_real, cell_stats, compare, sim_real_spearman, write_report, CLI
  scripts/
    __init__.py
    check_sim.py       # manual sim check + calibration
  tests/
    test_car_policy.py  test_nets.py  test_reward.py  test_sim_process.py  test_donkey_env.py
    test_cli.py  test_augment.py  test_wrappers.py  test_drivers.py  test_dataset.py
    test_collect.py  test_train_vae.py  test_train_bc.py  test_train_sac.py  test_metrics.py
    test_sim_bench.py  test_pilot_part.py  test_trial.py  test_schedule.py  test_report.py
```

---

### Task 1: Project scaffolding and environment

**Files:**
- Delete: `test_carla.py` (already deleted in the working tree; stage the deletion)
- Modify: `.gitignore`
- Create: `requirements.txt`, `pytest.ini`, `README.md`, `car/settings.json`
- Create (empty): `car/__init__.py`, `envs/__init__.py`, `collect/__init__.py`, `learn/__init__.py`, `bench/__init__.py`, `scripts/__init__.py`; directory `tests/`

**Interfaces:**
- Produces: importable empty packages `car`, `envs`, `collect`, `learn`, `bench`, `scripts`; pytest resolves imports from the repo root; `car/settings.json` with keys `throttle` (float), `car_dir` (str), `image_every` (int).

- [x] **Step 1: Stage the CARLA leftover deletion**

```bash
git rm test_carla.py
```

Expected: `git status` shows `deleted: test_carla.py` under "Changes to be committed".

- [x] **Step 2: Replace `.gitignore`**

```
# personal task list
todo.md

# python
__pycache__/
*.pyc
.venv/
.pytest_cache/

# large, reproducible outputs: driving data, models, results, recordings
data/
models/
results/
demo/
unitylog.txt
```

- [x] **Step 3: Write `requirements.txt`**

```
# Laptop only. The Pi needs nothing beyond donkeycar (numpy + Pillow).
# gym-donkeycar is pinned to a master commit: its PyPI release still uses the
# old `gym` API, master uses gymnasium.
gym-donkeycar @ git+https://github.com/tawnkramer/gym-donkeycar@a1f4ca6961e17a6929c1f2e883358a1edafb479d
gymnasium==1.3.0
stable-baselines3==2.9.0
torch==2.14.0
numpy>=2,<3
opencv-python-headless==5.0.0.93
pygame==2.6.1
tensorboard==2.21.0
scipy>=1.14,<2
matplotlib>=3.9,<4
pytest==9.1.1
```

- [x] **Step 4: Write `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [x] **Step 5: Write `car/settings.json`**

`throttle` is recalibrated on the real car later (todo Phase 6); keep this placeholder value until then.

```json
{
  "throttle": 0.3,
  "car_dir": "~/mycar",
  "image_every": 5
}
```

- [x] **Step 6: Write `README.md`**

````markdown
# Sim-to-Real Robustness Benchmark: RL vs Behavioral Cloning

Four steering policies (SAC and BC, each with and without domain
randomization) trained in the Donkey Car simulator, tested zero-shot under
six noise conditions in the simulator and on a real donkeycar.
Design: `docs/superpowers/specs/2026-09-16-sim2real-robustness-benchmark-design.md`.

## Laptop setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
pytest
```

Simulator (about 290 MB):

```bash
mkdir -p ~/donkey_sim
curl -L -o ~/donkey_sim/DonkeySimLinux.zip https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimLinux.zip
unzip -q ~/donkey_sim/DonkeySimLinux.zip -d ~/donkey_sim
find ~/donkey_sim -name '*.x86_64'
chmod +x <path printed by find>
export DONKEY_SIM_PATH=<path printed by find>
```

## Pipeline

Every sim command also takes `--env-name`, `--throttle`, `--cte-max`,
`--cam-fov`: use the same values everywhere.

```bash
python -m scripts.check_sim                                   # sim check + calibration
python -m collect.collect --mode scripted --n-frames 3000     # driving data (repeat, then --mode manual)
python -m learn.train_vae --out-dir models/vae_clean
python -m learn.train_vae --out-dir models/vae_dr --dr
python -m learn.train_bc --vae models/vae_clean/vae.pth --out models/bc_clean/policy.npz
python -m learn.train_bc --vae models/vae_dr/vae.pth --dr --out models/bc_dr/policy.npz
python -m learn.train_sac --encoder models/vae_clean/encoder.npz --run-dir models/sac_clean
python -m learn.train_sac --encoder models/vae_dr/encoder.npz --dr --run-dir models/sac_dr
python -m bench.sim_bench --conditions nominal                # model-quality gate
python -m bench.sim_bench                                     # full sim benchmark
python -m bench.schedule                                      # real-trial schedule -> car/schedule.csv
python -m bench.report                                        # results/report.md
```

On the Pi: `python -m car.trial <id>` (see `todo.md`).
If a crashed run left a simulator running: `pkill -f DonkeySim`.
````

- [x] **Step 7: Create packages**

```bash
mkdir -p car envs collect learn bench scripts tests
touch car/__init__.py envs/__init__.py collect/__init__.py learn/__init__.py bench/__init__.py scripts/__init__.py
```

- [x] **Step 8: Create the venv and install**

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Expected: completes without errors (torch is a large download).

- [x] **Step 9: Verify imports and GPU**

```bash
python -c "import gym_donkeycar, gymnasium, stable_baselines3, cv2, pygame, scipy, matplotlib, torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

Expected: `2.14.0 True [...]` with `sm_120` in the list. If CUDA is unavailable or `sm_120` is missing: `uv pip install --reinstall torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128`, then rerun. (CPU still works, just slower.)

- [x] **Step 10: Verify pytest configuration**

Run: `pytest`
Expected: `no tests ran` (exit code 5), no import or config errors.

- [x] **Step 11: Commit**

```bash
git add .gitignore requirements.txt pytest.ini README.md car envs collect learn bench scripts
git commit -m "chore: scaffold sim-to-real robustness benchmark"
```

---

### Task 2: Numpy policy inference (`car/policy.py`)

**Files:**
- Create: `car/policy.py`
- Test: `tests/test_car_policy.py`

**Interfaces:**
- Produces:
  - `CAMERA_SHAPE = (120, 160, 3)`, `CROP_TOP = 40`, `CLIP_MEAN = 2.0`
  - `preprocess(frame) -> np.ndarray` — `(120,160,3)` uint8 RGB → `(3,80,80)` float32 in [0, 1]; `ValueError` for any other shape.
  - `conv2d(x, weight, bias, stride=2, padding=1) -> np.ndarray` — one `C×H×W` input, same result as `torch.nn.functional.conv2d`.
  - `load_arrays(path) -> dict[str, np.ndarray]`
  - `Encoder(arrays)` — callable `(3,80,80) -> (latent_dim,)`; attribute `latent_dim`; `ValueError` if no `encoder.*` keys.
  - `Head(arrays)` — callable `obs (latent_dim+1,) -> float` in `[-tanh(2), tanh(2)]`; `ValueError` if no `head.*` keys.
  - `Policy(path)` — attributes `encoder`, `head`; `act(frame, prev_steer: float) -> float`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_car_policy.py
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_car_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'car.policy'`

- [x] **Step 3: Write the implementation**

```python
# car/policy.py
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
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_car_policy.py -v`
Expected: PASS (10 passed)

- [x] **Step 5: Commit**

```bash
git add car/policy.py tests/test_car_policy.py
git commit -m "feat: add numpy policy inference shared by sim and car"
```

---

### Task 3: Torch networks and export (`learn/nets.py`)

**Files:**
- Create: `learn/nets.py`
- Test: `tests/test_nets.py`

**Interfaces:**
- Consumes: `preprocess`, `CLIP_MEAN`, `Policy` (Task 2).
- Produces:
  - `LATENT_DIM = 32`, `HIDDEN = 64`
  - `Encoder(latent_dim=32)` (`nn.Module`): `features(x) -> (B, 12800)`, `forward(x) -> (B, latent_dim)`; `ValueError` unless input is `(B, 3, 80, 80)`; attribute `latent_dim`.
  - `ConvVAE(latent_dim=32)`: attribute `encoder: Encoder`, `latent_dim`; `decode(z) -> (B,3,80,80)` in [0,1]; `forward(x) -> (recon, mu, logvar)`.
  - `PolicyHead(obs_dim=33, hidden=64)`: `forward(obs (B, obs_dim)) -> (B, 1)`; submodules `latent_pi` (`Sequential` of Linear, ReLU, Linear, ReLU) and `mu` (Linear).
  - `frames_to_tensor(frames) -> torch.Tensor (B,3,80,80)`
  - `save_vae(model, path)`, `load_vae(path) -> ConvVAE` (eval mode, CPU)
  - `encoder_arrays(encoder) -> dict`, `head_arrays(head) -> dict` (numpy, keys prefixed `encoder.` / `head.`)
  - `save_npz(path, arrays: dict) -> None` (creates parent dirs)

- [x] **Step 1: Write the failing test**

```python
# tests/test_nets.py
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'learn.nets'`

- [x] **Step 3: Write the implementation**

```python
# learn/nets.py
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
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nets.py -v`
Expected: PASS (7 passed)

- [x] **Step 5: Commit**

```bash
git add learn/nets.py tests/test_nets.py
git commit -m "feat: add torch encoder, VAE and policy head with npz export"
```

---

### Task 4: Reward and simulator process management

**Files:**
- Create: `envs/reward.py`, `envs/sim_process.py`
- Test: `tests/test_reward.py`, `tests/test_sim_process.py`

**Interfaces:**
- Produces:
  - `CRASH_PENALTY = -10.0`; `is_hit(hit) -> bool`; `compute_reward(cte: float, hit, forward_vel: float, cte_max: float, sim_done: bool) -> tuple[float, bool]` returning `(reward, terminated)`.
  - `class SimDisconnectedError(RuntimeError)`
  - `call_with_timeout(fn, timeout: float, *args)` — returns `fn(*args)`; raises `SimDisconnectedError` after `timeout` s; re-raises `fn`'s exceptions.
  - `launch_sim(exe_path: str, port: int, timeout: float) -> subprocess.Popen` — starts `<exe> --port <port> --host 127.0.0.1 -logFile unitylog.txt`, returns once the port accepts connections; `SimDisconnectedError` if the process exits or the port doesn't open in time (killing it).
  - `kill_sim(proc: subprocess.Popen | None) -> None` — safe on `None` and exited processes.

Why the process code exists: gym-donkeycar's own launcher sleeps a fixed 5 s and loses the process handle on failure, orphaning a sim that keeps the port busy; its `observe()` waits forever when the sim dies.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_reward.py
import pytest

from envs.reward import CRASH_PENALTY, compute_reward, is_hit


def test_is_hit_treats_none_string_as_no_collision():
    assert is_hit("none") is False
    assert is_hit(None) is False
    assert is_hit("wall") is True


def test_lane_centre_gives_full_reward():
    assert compute_reward(cte=0.0, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=False) == (1.0, False)


def test_half_way_to_edge_gives_half_reward():
    reward, done = compute_reward(cte=-1.0, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=False)
    assert reward == pytest.approx(0.5)
    assert not done


def test_cte_spike_without_sim_done_is_not_terminal():
    assert compute_reward(cte=9.0, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=False) == (0.0, False)


@pytest.mark.parametrize("forward_vel", [0.0, -2.0])
def test_not_moving_forward_earns_nothing(forward_vel):
    assert compute_reward(cte=0.0, hit="none", forward_vel=forward_vel, cte_max=2.0, sim_done=False) == (0.0, False)


def test_collision_ends_episode_with_penalty():
    assert compute_reward(cte=0.0, hit="wall", forward_vel=1.0, cte_max=2.0, sim_done=False) == (CRASH_PENALTY, True)


def test_sim_done_ends_episode_with_penalty():
    assert compute_reward(cte=2.5, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=True) == (CRASH_PENALTY, True)
```

```python
# tests/test_sim_process.py
import socket
import sys
import time

import pytest

from envs.sim_process import SimDisconnectedError, call_with_timeout, kill_sim, launch_sim


def test_call_with_timeout_returns_value():
    assert call_with_timeout(lambda x: x * 2, 1.0, 21) == 42


def test_call_with_timeout_raises_on_hang():
    start = time.monotonic()
    with pytest.raises(SimDisconnectedError):
        call_with_timeout(time.sleep, 0.2, 5)
    assert time.monotonic() - start < 2


def test_call_with_timeout_reraises_errors():
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        call_with_timeout(boom, 1.0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fake_sim(tmp_path, body: str):
    """Write an executable script that accepts the real sim's CLI args."""
    script = tmp_path / "fake_sim"
    script.write_text(
        f"#!{sys.executable}\n"
        "import socket, sys, time\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        f"{body}\n"
    )
    script.chmod(0o755)
    return str(script)


def test_launch_sim_waits_for_port(tmp_path):
    exe = _fake_sim(
        tmp_path,
        "time.sleep(0.5)\n"
        "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "s.bind(('127.0.0.1', port)); s.listen()\n"
        "time.sleep(60)",
    )
    proc = launch_sim(exe, _free_port(), timeout=10)
    try:
        assert proc.poll() is None
    finally:
        kill_sim(proc)
    assert proc.poll() is not None


def test_launch_sim_reports_early_exit(tmp_path):
    exe = _fake_sim(tmp_path, "sys.exit(3)")
    with pytest.raises(SimDisconnectedError, match="code 3"):
        launch_sim(exe, _free_port(), timeout=10)


def test_launch_sim_times_out_and_kills(tmp_path):
    exe = _fake_sim(tmp_path, "time.sleep(60)")
    with pytest.raises(SimDisconnectedError, match="did not open port"):
        launch_sim(exe, _free_port(), timeout=1.5)


def test_kill_sim_accepts_none():
    kill_sim(None)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reward.py tests/test_sim_process.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write `envs/reward.py`**

```python
# envs/reward.py
CRASH_PENALTY = -10.0


def is_hit(hit) -> bool:
    """gym-donkeycar reports a collision as the name of the object hit and the
    string "none" otherwise. "none" is truthy, so never test it for truthiness."""
    return hit is not None and hit != "none"


def compute_reward(cte: float, hit, forward_vel: float, cte_max: float, sim_done: bool) -> tuple[float, bool]:
    """Returns (reward, terminated) for steering-only lane following.

    Throttle is fixed, so reward only asks: how close to the lane centre?
      reward = 1 at the centre, falling linearly to 0 at cte_max
      0 when not moving forward (after a spin the car may face backwards)
    A collision, or the sim's own game-over (|cte| > max_cte, missed
    checkpoint, disqualification), ends the episode with CRASH_PENALTY.
    Off-lane is deliberately not re-checked here: the sim ignores the CTE
    spikes it reports just after a reset, and re-checking would not.
    """
    if sim_done or is_hit(hit):
        return CRASH_PENALTY, True
    if forward_vel <= 0.0:
        return 0.0, False
    return 1.0 - min(abs(cte) / cte_max, 1.0), False
```

- [x] **Step 4: Write `envs/sim_process.py`**

```python
# envs/sim_process.py
import socket
import subprocess
import threading
import time


class SimDisconnectedError(RuntimeError):
    """The simulator failed to start or stopped answering.

    Recovery is always the same: close the env and build a new one.
    """


def call_with_timeout(fn, timeout: float, *args):
    """Run fn(*args) in a daemon thread. Raise SimDisconnectedError if it
    hasn't returned after timeout seconds; re-raise anything fn raised.

    gym-donkeycar busy-waits forever for the next frame when the sim dies,
    so a timeout is the only way to notice. The stuck thread is a daemon,
    so it can't keep the process alive at exit.
    """
    result = {}

    def target():
        try:
            result["value"] = fn(*args)
        except BaseException as exc:  # handed back to the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise SimDisconnectedError(f"simulator did not respond within {timeout}s")
    if "error" in result:
        raise result["error"]
    return result["value"]


def launch_sim(exe_path: str, port: int, timeout: float) -> subprocess.Popen:
    """Start the simulator binary and wait until it accepts TCP connections.

    We own the process (rather than letting gym-donkeycar start it) so a
    failed start never leaves an orphaned sim holding the port.
    """
    proc = subprocess.Popen(
        [exe_path, "--port", str(port), "--host", "127.0.0.1", "-logFile", "unitylog.txt"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SimDisconnectedError(f"simulator exited during startup with code {proc.returncode}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
            return proc
        except OSError:
            time.sleep(0.5)
    kill_sim(proc)
    raise SimDisconnectedError(f"simulator did not open port {port} within {timeout}s")


def kill_sim(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.kill()
    proc.wait(timeout=10)
```

- [x] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_reward.py tests/test_sim_process.py -v`
Expected: PASS (15 passed)

- [x] **Step 6: Commit**

```bash
git add envs/reward.py envs/sim_process.py tests/test_reward.py tests/test_sim_process.py
git commit -m "feat: add lane reward and simulator process management"
```

---

### Task 5: Steering-only `DonkeyLaneEnv` and shared CLI flags

**Files:**
- Create: `envs/donkey_env.py`, `envs/cli.py`
- Test: `tests/test_donkey_env.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `compute_reward` (Task 4); `SimDisconnectedError`, `call_with_timeout`, `launch_sim`, `kill_sim` (Task 4); `CAMERA_SHAPE` (Task 2).
- Produces:
  - `_make_underlying_env(env_name: str, conf: dict)` — module-level so tests can monkeypatch it.
  - `DonkeyLaneEnv(exe_path: str, env_name: str = "donkey-generated-track-v0", port: int = 9091, cte_max: float = 2.0, max_episode_steps: int = 2000, throttle: float = 0.25, steer_limit: float = 1.0, cam_fov: int = 0, step_timeout: float = 10.0, startup_timeout: float = 120.0)`. `exe_path="remote"` skips launching. `action_space = Box(-1, 1, (1,), float32)` (steering). `observation_space = Box(0, 255, CAMERA_SHAPE, uint8)`. Public mutable attribute `throttle` (sim throttle sent every step). `reset(*, seed=None, options=None) -> (frame, info)`, `step(action) -> (frame, reward, terminated, truncated, info)`, `close()` (idempotent). Raises `FileNotFoundError` for a missing binary and `SimDisconnectedError` when the sim can't be started/reached or stops answering. When `cam_fov > 0`, `conf["cam_config"] = {"img_w": 160, "img_h": 120, "fov": cam_fov}`.
  - `add_env_args(parser)` adds `--exe-path` (default `$DONKEY_SIM_PATH`), `--port`, `--env-name`, `--cte-max`, `--max-episode-steps`, `--throttle`, `--cam-fov`; `env_kwargs_from_args(args) -> dict` for `DonkeyLaneEnv`, exiting with a message if no sim path.

- [x] **Step 1: Write the failing tests**

The fake mirrors gym-donkeycar's real API — 5-tuple `step`, `hit` as the string `"none"`.

```python
# tests/test_donkey_env.py
import time

import numpy as np
import pytest

import envs.donkey_env as donkey_env
from envs.donkey_env import DonkeyLaneEnv
from envs.reward import CRASH_PENALTY
from envs.sim_process import SimDisconnectedError


class FakeUnderlyingEnv:
    """Stands in for the gym_donkeycar env: 5-tuple step, hit is "none" when clean."""

    def __init__(self):
        self.infos = []
        self.last_action = None
        self.closed = False
        self.hang = False

    def reset(self):
        return np.zeros((120, 160, 3), dtype=np.uint8), {"cte": 0.0, "hit": "none"}

    def step(self, action):
        if self.hang:
            time.sleep(60)
        self.last_action = np.asarray(action)
        info = self.infos.pop(0) if self.infos else {"cte": 0.0, "hit": "none", "forward_vel": 1.0}
        done = info.pop("_done", False)
        return np.zeros((120, 160, 3), dtype=np.uint8), 0.0, done, False, info

    def close(self):
        self.closed = True


@pytest.fixture
def fake():
    return FakeUnderlyingEnv()


@pytest.fixture
def captured():
    return {}


@pytest.fixture
def make_env(monkeypatch, fake, captured):
    def factory(env_name, conf):
        captured.update(env_name=env_name, conf=conf)
        return fake

    monkeypatch.setattr(donkey_env, "_make_underlying_env", factory)
    return lambda **kwargs: DonkeyLaneEnv(exe_path="remote", **kwargs)


def test_action_space_is_steering_only(make_env):
    assert make_env().action_space.shape == (1,)


def test_missing_binary_fails_fast():
    with pytest.raises(FileNotFoundError):
        DonkeyLaneEnv(exe_path="/definitely/not/here")


def test_reset_returns_raw_frame(make_env):
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (120, 160, 3) and obs.dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_step_scales_steering_and_sends_fixed_throttle(make_env, fake):
    env = make_env(steer_limit=0.8, throttle=0.3)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(np.array([0.5]))
    assert np.allclose(fake.last_action, [0.4, 0.3])
    assert reward == pytest.approx(1.0)
    assert not terminated and not truncated


def test_steering_is_clipped(make_env, fake):
    env = make_env()
    env.reset()
    env.step(np.array([5.0]))
    assert fake.last_action[0] == pytest.approx(1.0)


def test_throttle_attribute_can_change_between_episodes(make_env, fake):
    env = make_env(throttle=0.3)
    env.throttle = 0.1
    env.reset()
    env.step(np.zeros(1))
    assert fake.last_action[1] == pytest.approx(0.1)


def test_hit_none_string_does_not_terminate(make_env, fake):
    env = make_env()
    env.reset()
    fake.infos = [{"cte": 0.0, "hit": "none", "forward_vel": 1.0}]
    assert env.step(np.zeros(1))[2] is False


def test_collision_terminates(make_env, fake):
    env = make_env()
    env.reset()
    fake.infos = [{"cte": 0.0, "hit": "wall", "forward_vel": 1.0}]
    _, reward, terminated, truncated, _ = env.step(np.zeros(1))
    assert terminated is True and truncated is False
    assert reward == CRASH_PENALTY


def test_sim_game_over_terminates(make_env, fake):
    env = make_env()
    env.reset()
    fake.infos = [{"cte": 2.5, "hit": "none", "forward_vel": 1.0, "_done": True}]
    assert env.step(np.zeros(1))[2] is True


def test_truncates_at_max_episode_steps(make_env):
    env = make_env(max_episode_steps=3)
    env.reset()
    assert [env.step(np.zeros(1))[3] for _ in range(3)] == [False, False, True]
    env.reset()
    assert env.step(np.zeros(1))[3] is False


def test_hung_sim_raises_disconnected(make_env, fake):
    env = make_env(step_timeout=0.2)
    env.reset()
    fake.hang = True
    with pytest.raises(SimDisconnectedError):
        env.step(np.zeros(1))


def test_connection_failure_becomes_disconnected_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise Exception("Could not connect to server. Is it running?")

    monkeypatch.setattr(donkey_env, "_make_underlying_env", refuse)
    with pytest.raises(SimDisconnectedError, match="could not connect"):
        DonkeyLaneEnv(exe_path="remote")


def test_close_is_idempotent(make_env, fake):
    env = make_env()
    env.close()
    env.close()
    assert fake.closed


def test_cam_fov_is_sent_only_when_set(make_env, captured):
    make_env()
    assert "cam_config" not in captured["conf"]
    make_env(cam_fov=49)
    assert captured["conf"]["cam_config"] == {"img_w": 160, "img_h": 120, "fov": 49}
```

```python
# tests/test_cli.py
import argparse

import pytest

from envs.cli import add_env_args, env_kwargs_from_args


def _parse(argv, monkeypatch, env_var=None):
    if env_var is None:
        monkeypatch.delenv("DONKEY_SIM_PATH", raising=False)
    else:
        monkeypatch.setenv("DONKEY_SIM_PATH", env_var)
    parser = argparse.ArgumentParser()
    add_env_args(parser)
    return parser.parse_args(argv)


def test_exe_path_falls_back_to_env_var(monkeypatch):
    assert env_kwargs_from_args(_parse([], monkeypatch, "/sims/donkey"))["exe_path"] == "/sims/donkey"


def test_missing_exe_path_exits_with_message(monkeypatch):
    with pytest.raises(SystemExit, match="DONKEY_SIM_PATH"):
        env_kwargs_from_args(_parse([], monkeypatch))


def test_flags_reach_kwargs(monkeypatch):
    args = _parse(["--exe-path", "remote", "--throttle", "0.3", "--cte-max", "1.5", "--cam-fov", "49"], monkeypatch)
    kwargs = env_kwargs_from_args(args)
    assert kwargs["throttle"] == 0.3
    assert kwargs["cte_max"] == 1.5
    assert kwargs["cam_fov"] == 49
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_donkey_env.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write `envs/donkey_env.py`**

```python
# envs/donkey_env.py
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from car.policy import CAMERA_SHAPE
from envs.reward import compute_reward
from envs.sim_process import SimDisconnectedError, call_with_timeout, kill_sim, launch_sim


def _make_underlying_env(env_name: str, conf: dict):
    """Thin factory around gym.make so tests can monkeypatch it."""
    import gym_donkeycar  # noqa: F401  (registers the donkey-* envs)

    return gym.make(env_name, conf=conf).unwrapped


class DonkeyLaneEnv(gym.Env):
    """Steering-only lane-following wrapper around gym-donkeycar.

    Action: [steering] in [-1, 1], scaled by steer_limit.
    Throttle: the fixed value in self.throttle, sent every step. It is a
    public attribute so the benchmark can lower it for the low_battery condition.
    Observation: the raw uint8 RGB camera frame.
    terminated: collision / off lane / sim game-over.
    truncated: max_episode_steps reached.
    Raises SimDisconnectedError if the sim won't start or stops answering;
    close() this env and build a new one.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        exe_path: str,
        env_name: str = "donkey-generated-track-v0",
        port: int = 9091,
        cte_max: float = 2.0,
        max_episode_steps: int = 2000,
        throttle: float = 0.25,
        steer_limit: float = 1.0,
        cam_fov: int = 0,
        step_timeout: float = 10.0,
        startup_timeout: float = 120.0,
    ):
        super().__init__()
        if exe_path != "remote" and not os.path.isfile(exe_path):
            raise FileNotFoundError(
                f"simulator binary not found: {exe_path!r} (pass 'remote' to use a sim you started yourself)"
            )
        self.cte_max = cte_max
        self.max_episode_steps = max_episode_steps
        self.throttle = throttle
        self.steer_limit = steer_limit
        self.step_timeout = step_timeout
        self.observation_space = spaces.Box(0, 255, shape=CAMERA_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._steps = 0
        self._underlying = None
        self._proc = None
        if exe_path != "remote":
            self._proc = launch_sim(exe_path, port, startup_timeout)
        # "remote": gym-donkeycar must not start a sim; we already did (or the user did).
        # max_cte makes the sim's own game-over check use our lane width.
        conf = {"exe_path": "remote", "port": port, "max_cte": cte_max, "start_delay": 0.0}
        if cam_fov > 0:
            conf["cam_config"] = {"img_w": CAMERA_SHAPE[1], "img_h": CAMERA_SHAPE[0], "fov": cam_fov}
        try:
            self._underlying = call_with_timeout(_make_underlying_env, startup_timeout, env_name, conf)
        except SimDisconnectedError:
            self.close()
            raise
        except Exception as exc:
            # gym-donkeycar raises a bare Exception when it can't connect.
            self.close()
            raise SimDisconnectedError(f"could not connect to simulator: {exc}") from exc

    def _guarded(self, fn, *args):
        try:
            return call_with_timeout(fn, self.step_timeout, *args)
        except (ConnectionError, OSError) as exc:
            raise SimDisconnectedError(str(exc)) from exc

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        frame, info = self._guarded(self._underlying.reset)
        self._steps = 0
        return np.asarray(frame, dtype=np.uint8), info

    def step(self, action):
        steer = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        sim_action = np.array([steer * self.steer_limit, self.throttle], dtype=np.float32)
        frame, _sim_reward, sim_done, _sim_truncated, info = self._guarded(self._underlying.step, sim_action)
        self._steps += 1
        reward, terminated = compute_reward(
            cte=info.get("cte", 0.0),
            hit=info.get("hit", "none"),
            forward_vel=info.get("forward_vel", 0.0),
            cte_max=self.cte_max,
            sim_done=sim_done,
        )
        truncated = not terminated and self._steps >= self.max_episode_steps
        return np.asarray(frame, dtype=np.uint8), reward, terminated, truncated, info

    def close(self):
        # Kill the sim first: that closes its socket, which unblocks
        # gym-donkeycar's network thread so its own close() can finish.
        kill_sim(self._proc)
        self._proc = None
        if self._underlying is not None:
            try:
                call_with_timeout(self._underlying.close, 10.0)
            except Exception:
                pass  # best effort: the sim is already gone
            self._underlying = None
```

- [x] **Step 4: Write `envs/cli.py`**

```python
# envs/cli.py
"""Command-line flags shared by every script that builds a DonkeyLaneEnv.
Use identical values for collection, training and the benchmark."""
import argparse
import os


def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exe-path",
        default=os.environ.get("DONKEY_SIM_PATH"),
        help="simulator binary, or 'remote' for a sim you started yourself (default: $DONKEY_SIM_PATH)",
    )
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--env-name", default="donkey-generated-track-v0")
    parser.add_argument("--cte-max", type=float, default=2.0, help="lane half-width; beyond it the episode ends")
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--throttle", type=float, default=0.25, help="fixed sim throttle")
    parser.add_argument("--cam-fov", type=int, default=0, help="sim camera FOV in degrees; 0 keeps the sim default")


def env_kwargs_from_args(args: argparse.Namespace) -> dict:
    if not args.exe_path:
        raise SystemExit("error: pass --exe-path or set DONKEY_SIM_PATH")
    return {
        "exe_path": args.exe_path,
        "port": args.port,
        "env_name": args.env_name,
        "cte_max": args.cte_max,
        "max_episode_steps": args.max_episode_steps,
        "throttle": args.throttle,
        "cam_fov": args.cam_fov,
    }
```

- [x] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_donkey_env.py tests/test_cli.py -v`
Expected: PASS (17 passed)

- [x] **Step 6: Commit**

```bash
git add envs/donkey_env.py envs/cli.py tests/test_donkey_env.py tests/test_cli.py
git commit -m "feat: add steering-only DonkeyLaneEnv and shared sim flags"
```

---

### Task 6: Simulator check and calibration script (manual)

**Files:**
- Create: `scripts/check_sim.py`

**Interfaces:**
- Consumes: `DonkeyLaneEnv`, `add_env_args`, `env_kwargs_from_args` (Task 5).
- Produces: nothing importable. Manual calibration values recorded in `todo.md`: chosen track (`--env-name`), CTE sign (`--cte-sign`), lane half-width (`--cte-max`), sim throttle (`--throttle`), camera FOV (`--cam-fov`).

**Prerequisite (the user does this):** the simulator is unzipped and `DONKEY_SIM_PATH` points at the executable. If not, stop and ask the user.

- [x] **Step 1: Write `scripts/check_sim.py`**

```python
# scripts/check_sim.py
"""Manual check: launch the simulator through DonkeyLaneEnv and drive with a
constant steering value, printing what the sim reports.

Uses:
  * confirm the sim starts, connects and closes cleanly
  * confirm the info keys the code relies on (cte, hit, forward_vel, lap_count, pos)
  * CTE sign: --steer 0.5 and watch whether cte goes up or down
  * lane half-width: note |cte| when the wheels cross the lane edge
  * throttle: find the lowest --throttle that drives smoothly
  * --save-frame out.png saves the camera frame at the last step, to compare
    the sim view with a real camera frame (track choice, --cam-fov, camera tilt)
"""
import argparse

import cv2
import numpy as np

from envs.cli import add_env_args, env_kwargs_from_args
from envs.donkey_env import DonkeyLaneEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(parser)
    parser.add_argument("--steer", type=float, default=0.0, help="constant steering in [-1, 1]")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-frame", default=None, help="write the last camera frame to this PNG")
    args = parser.parse_args()

    env = DonkeyLaneEnv(**env_kwargs_from_args(args))
    frame = None
    try:
        frame, info = env.reset()
        print("Connected. Frame:", frame.shape, frame.dtype)
        print("Info keys:", sorted(info.keys()))
        action = np.array([args.steer], dtype=np.float32)
        for step in range(1, args.steps + 1):
            frame, reward, terminated, truncated, info = env.step(action)
            if step % 10 == 0 or terminated or truncated:
                print(
                    f"step {step:4d}  cte {info.get('cte', 0.0):+.3f}  hit {info.get('hit')!r}  "
                    f"forward_vel {info.get('forward_vel', 0.0):+.2f}  lap_count {info.get('lap_count')}  "
                    f"pos {info.get('pos')}  reward {reward:+.3f}"
                )
            if terminated or truncated:
                print("Episode ended:", "terminated" if terminated else "truncated")
                frame, info = env.reset()
    finally:
        env.close()
    if args.save_frame and frame is not None:
        cv2.imwrite(args.save_frame, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Saved", args.save_frame)
    print("Closed cleanly.")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Manual — connection check**

Run: `python -m scripts.check_sim --steps 100`
Expected: a sim window opens and the car drives forward; `Info keys:` includes `cte`, `forward_vel`, `hit`, `lap_count`, `last_lap_time`, `pos`; `hit` prints `'none'` while driving cleanly; the script ends with `Closed cleanly.`; afterwards `pgrep -fa DonkeySim` prints nothing.
If a key is missing or renamed, stop: `envs/reward.py`, `envs/donkey_env.py` and `bench/sim_bench.py` rely on these names.

- [x] **Step 3: Manual — track choice**

For each of `donkey-generated-track-v0` and `donkey-warehouse-v0`:
`python -m scripts.check_sim --env-name <track> --steps 60 --save-frame data/track_<track>.png`
(create `data/` first: `mkdir -p data`). Open both PNGs. Pick the track that looks most like a tape line track on a floor and whose `lap_count` increases when the car completes a lap. Record it in `todo.md`; pass it as `--env-name` to every later command.

- [x] **Step 4: Manual — CTE sign, lane half-width, throttle**

- CTE sign: `python -m scripts.check_sim --env-name <track> --steer 0.5 --steps 80 --cte-max 8`. `cte` grows more positive under right steer → sign `+1`, more negative → `-1`.
- Lane half-width: in that run note `|cte|` when the wheels visibly cross the lane edge; repeat with `--steer -0.5`; take the smaller value → `--cte-max`.
- Throttle: `python -m scripts.check_sim --env-name <track> --throttle 0.15 --steps 60`, then 0.2, 0.25, 0.3. Pick a value where the car moves steadily and slowly → `--throttle`.

Record all three in `todo.md`.

- [x] **Step 5: Commit**

```bash
git add scripts/check_sim.py
git commit -m "feat: add simulator check and calibration script"
```

---
### Task 7: Image augmentation (training DR + held-out evaluation perturbations)

**Files:**
- Create: `envs/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Produces: augmenters sharing one interface — constructor `(rng: np.random.Generator)`, `new_episode() -> None` (samples per-episode settings; the constructor calls it once), `__call__(frame: uint8 (120,160,3), t: int = 0) -> uint8 (120,160,3)` (`t` = step within the episode; never modifies its input).
  - `DomainRandomizer` (training), `LowLight`, `MovingShadow`, `Distractors` (evaluation)
  - `AUGMENTERS = {"dr": DomainRandomizer, "low_light": LowLight, "shadows": MovingShadow, "distractors": Distractors}`
  - `make_augmenter(name: str | None, rng) -> augmenter | None` (`None` → `None`; unknown name → `KeyError`)

Training ranges and evaluation strengths are disjoint on purpose (spec, "Domain randomization").

- [x] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import numpy as np
import pytest

from car.policy import CAMERA_SHAPE
from envs.augment import AUGMENTERS, DomainRandomizer, Distractors, LowLight, MovingShadow, make_augmenter


def _frame(value=150):
    return np.full(CAMERA_SHAPE, value, dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(AUGMENTERS))
def test_shape_dtype_and_input_untouched(name):
    frame = np.random.default_rng(0).integers(0, 256, size=CAMERA_SHAPE, dtype=np.uint8)
    original = frame.copy()
    out = make_augmenter(name, np.random.default_rng(1))(frame, 3)
    assert out.shape == CAMERA_SHAPE and out.dtype == np.uint8
    assert np.array_equal(frame, original)


def test_make_augmenter_none_and_unknown():
    assert make_augmenter(None, np.random.default_rng(0)) is None
    with pytest.raises(KeyError):
        make_augmenter("fog", np.random.default_rng(0))


def test_domain_randomizer_is_deterministic_for_a_seed():
    a = DomainRandomizer(np.random.default_rng(7))(_frame())
    b = DomainRandomizer(np.random.default_rng(7))(_frame())
    assert np.array_equal(a, b)


def test_domain_randomizer_changes_between_episodes():
    dr = DomainRandomizer(np.random.default_rng(0))
    first = dr(_frame()).astype(int)
    dr.new_episode()
    assert np.abs(dr(_frame()).astype(int) - first).mean() > 1.0


def test_training_brightness_never_reaches_low_light_strength():
    dr = DomainRandomizer(np.random.default_rng(0))
    values = []
    for _ in range(500):
        dr.new_episode()
        values.append(dr.brightness)
    assert min(values) >= 0.6 > LowLight.BRIGHTNESS


def test_low_light_darkens():
    out = LowLight(np.random.default_rng(0))(_frame(150))
    assert out.mean() < 0.5 * 150


def test_moving_shadow_moves_over_time():
    shadow = MovingShadow(np.random.default_rng(0))
    darkened = [(shadow(_frame(200), t) < 200).sum() for t in range(0, 120, 4)]
    assert max(darkened) > 0
    assert not np.array_equal(shadow(_frame(200), 0), shadow(_frame(200), 10))


def test_distractors_fixed_within_episode_and_change_between():
    d = Distractors(np.random.default_rng(0))
    first = d(_frame(), 0)
    assert np.array_equal(first, d(_frame(), 50))
    assert (first != 150).any()
    d.new_episode()
    assert not np.array_equal(first, d(_frame(), 0))
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_augment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'envs.augment'`

- [x] **Step 3: Write the implementation**

```python
# envs/augment.py
"""Image-space domain randomization (training) and held-out perturbations (evaluation).

Every augmenter has the same interface:
    aug = Augmenter(rng)       # rng: np.random.Generator
    aug.new_episode()          # sample per-episode settings (also done once by the constructor)
    aug(frame, t) -> frame     # uint8 120x160x3 RGB in and out; t = step within the episode

Training ranges and evaluation strengths never overlap, so a model trained
with DR can't pass an evaluation condition just by having seen it:
  low light  eval brightness 0.35   < training minimum 0.6
  shadows    eval: moving band, darkness 0.3; training: static polygons, darkness >= 0.5
  distractors eval only
"""
import cv2
import numpy as np


def _polygon_mask(shape, points) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask.astype(bool)


class DomainRandomizer:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.new_episode()

    def new_episode(self) -> None:
        r = self.rng
        self.brightness = r.uniform(0.6, 1.4)
        self.contrast = r.uniform(0.7, 1.3)
        self.saturation = r.uniform(0.7, 1.3)
        self.hue_shift = r.uniform(-8.0, 8.0)  # OpenCV hue units (0-180)
        self.shadow_polygons = [
            r.uniform((0, 0), (160, 120), size=(int(r.integers(3, 6)), 2)) for _ in range(int(r.integers(0, 3)))
        ]
        self.shadow_darkness = r.uniform(0.5, 0.8)
        self.noise_std = r.uniform(0.0, 8.0)
        self.blur = bool(r.random() < 0.3)

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self.hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * self.saturation, 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        mean = x.mean()
        x = ((x - mean) * self.contrast + mean) * self.brightness
        for polygon in self.shadow_polygons:
            x[_polygon_mask(x.shape, polygon)] *= self.shadow_darkness
        x += self.rng.normal(0.0, self.noise_std, x.shape)
        out = np.clip(x, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(out, (3, 3), 0) if self.blur else out


class LowLight:
    """Dim scene: darker image with more sensor noise."""

    BRIGHTNESS = 0.35
    NOISE_STD = 6.0

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def new_episode(self) -> None:
        pass

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32) * self.BRIGHTNESS
        x += self.rng.normal(0.0, self.NOISE_STD, x.shape)
        return np.clip(x, 0, 255).astype(np.uint8)


class MovingShadow:
    """A dark diagonal band sweeping sideways across the image, like the
    shadow of something moving past the track."""

    DARKNESS = 0.3
    WIDTH = 30  # pixels
    SPEED = 4  # pixels per step

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.new_episode()

    def new_episode(self) -> None:
        self.offset = int(self.rng.integers(0, 1000))

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32)
        h, w = x.shape[:2]
        period = w + h + 2 * self.WIDTH
        left = (self.offset + self.SPEED * t) % period - self.WIDTH - h
        ys, xs = np.mgrid[0:h, 0:w]
        d = xs - ys - left
        x[(d >= 0) & (d < self.WIDTH)] *= self.DARKNESS
        return x.astype(np.uint8)


class Distractors:
    """Three saturated rectangles (objects beside the track), fixed per episode."""

    COLOURS = ((220, 30, 30), (30, 60, 220), (240, 140, 0))

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.new_episode()

    def new_episode(self) -> None:
        self.rects = []
        for colour in self.COLOURS:
            w, h = (int(v) for v in self.rng.integers(12, 26, size=2))
            x = int(self.rng.integers(0, 160 - w))
            y = int(self.rng.integers(50, 120 - h))  # below the cropped top rows
            self.rects.append((x, y, w, h, colour))

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        out = np.array(frame, dtype=np.uint8, copy=True)
        for x, y, w, h, colour in self.rects:
            out[y : y + h, x : x + w] = colour
        return out


AUGMENTERS = {"dr": DomainRandomizer, "low_light": LowLight, "shadows": MovingShadow, "distractors": Distractors}


def make_augmenter(name: str | None, rng: np.random.Generator):
    return None if name is None else AUGMENTERS[name](rng)
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_augment.py -v`
Expected: PASS (11 passed)

- [x] **Step 5: Commit**

```bash
git add envs/augment.py tests/test_augment.py
git commit -m "feat: add domain randomization and held-out image perturbations"
```

---

### Task 8: Env wrappers (`ActuatorLag`, `LatentEnv`)

**Files:**
- Create: `envs/wrappers.py`
- Test: `tests/test_wrappers.py`

**Interfaces:**
- Consumes: `preprocess` (Task 2); any gymnasium env with a `(1,)` steering action (Task 5); augmenter interface (Task 7).
- Produces:
  - `ActuatorLag(env, delay_steps: int = 0)` — each steering action reaches `env` `delay_steps` steps late (zeros first). Public attribute `delay_steps`, applied at the next `reset`.
  - `LatentEnv(env, encoder, augmenter=None)` — `encoder` is callable `(3,80,80) float32 -> (latent_dim,)` with attribute `latent_dim` (e.g. `car.policy.Encoder`). Observation `Box(-inf, inf, (latent_dim+1,), float32)` = `[encoder(preprocess(augmented frame)), previous steering command]`. The previous command is the clipped action given to `LatentEnv.step` (0 after reset). `augmenter(frame, t)` receives the step index `t` (0 at reset); `augmenter.new_episode()` is called on every reset. Public attributes `encoder` and `augmenter` may be swapped between episodes.

- [x] **Step 1: Write the failing test**

```python
# tests/test_wrappers.py
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from car.policy import CAMERA_SHAPE
from envs.wrappers import ActuatorLag, LatentEnv


class FrameEnv(gym.Env):
    """Frames whose pixel value is the step count; records actions received."""

    def __init__(self):
        self.observation_space = spaces.Box(0, 255, shape=CAMERA_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.actions = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(CAMERA_SHAPE, dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(float(np.asarray(action).reshape(-1)[0]))
        self.steps += 1
        return np.full(CAMERA_SHAPE, self.steps, dtype=np.uint8), 1.0, False, False, {"cte": 0.1}


class MeanEncoder:
    """latent = [mean pixel value (0-1), 0, 0, 0]"""

    latent_dim = 4

    def __call__(self, x):
        return np.array([x.mean(), 0.0, 0.0, 0.0], dtype=np.float32)


class RecordingAugmenter:
    def __init__(self):
        self.episodes = 0
        self.ts = []

    def new_episode(self):
        self.episodes += 1

    def __call__(self, frame, t=0):
        self.ts.append(t)
        return np.full_like(frame, 255)


def test_actuator_lag_delays_steering():
    inner = FrameEnv()
    env = ActuatorLag(inner, delay_steps=2)
    env.reset()
    for a in (0.1, 0.2, 0.3, 0.4):
        env.step(np.array([a], dtype=np.float32))
    assert inner.actions == pytest.approx([0.0, 0.0, 0.1, 0.2])


def test_actuator_lag_zero_is_passthrough_and_delay_applies_on_reset():
    inner = FrameEnv()
    env = ActuatorLag(inner)
    env.reset()
    env.step(np.array([0.5], dtype=np.float32))
    env.delay_steps = 1
    env.reset()
    env.step(np.array([0.7], dtype=np.float32))
    assert inner.actions == pytest.approx([0.5, 0.0])


def test_latent_env_observation_is_latent_plus_previous_steer():
    env = LatentEnv(FrameEnv(), MeanEncoder())
    assert env.observation_space.shape == (5,)
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    assert obs[-1] == 0.0
    obs, reward, terminated, truncated, info = env.step(np.array([3.0]))
    assert obs[-1] == pytest.approx(1.0)  # clipped command
    assert obs[0] == pytest.approx(1 / 255, abs=1e-6)  # frame after step 1 has pixel value 1
    assert reward == 1.0 and info == {"cte": 0.1}
    obs, _ = env.reset()
    assert obs[-1] == 0.0


def test_latent_env_sends_clipped_one_element_action():
    inner = FrameEnv()
    env = LatentEnv(inner, MeanEncoder())
    env.reset()
    env.step(np.array([[-4.0]]))
    assert inner.actions == [-1.0]


def test_latent_env_applies_augmenter_with_step_index():
    aug = RecordingAugmenter()
    env = LatentEnv(FrameEnv(), MeanEncoder(), augmenter=aug)
    obs, _ = env.reset()
    env.step(np.zeros(1))
    env.step(np.zeros(1))
    assert aug.episodes == 1
    assert aug.ts == [0, 1, 2]
    assert obs[0] == pytest.approx(1.0)  # augmenter made the frame white
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wrappers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'envs.wrappers'`

- [x] **Step 3: Write the implementation**

```python
# envs/wrappers.py
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from car.policy import preprocess


class ActuatorLag(gym.Wrapper):
    """Delays each steering command by delay_steps env steps (0 = no delay).
    The simulator's stand-in for a slow servo on a low battery. Changes to
    delay_steps take effect at the next reset."""

    def __init__(self, env: gym.Env, delay_steps: int = 0):
        super().__init__(env)
        self.delay_steps = delay_steps
        self._queue = deque()

    def reset(self, **kwargs):
        self._queue = deque([np.zeros(1, dtype=np.float32)] * self.delay_steps)
        return self.env.reset(**kwargs)

    def step(self, action):
        self._queue.append(np.asarray(action, dtype=np.float32).reshape(1))
        return self.env.step(self._queue.popleft())


class LatentEnv(gym.Wrapper):
    """Observation = [encoder(preprocess(frame)), previous steering command].

    This is exactly what the real car computes (car/policy.py::Policy.act),
    so a policy trained or evaluated here sees the same inputs on the car.
    encoder: callable 3x80x80 -> (latent_dim,) with a latent_dim attribute.
    augmenter: optional envs.augment augmenter applied to each raw frame.
    Both are public and may be swapped between episodes (the benchmark does).
    """

    def __init__(self, env: gym.Env, encoder, augmenter=None):
        super().__init__(env)
        self.encoder = encoder
        self.augmenter = augmenter
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(encoder.latent_dim + 1,), dtype=np.float32)
        self._t = 0
        self._prev = 0.0

    def _observe(self, frame) -> np.ndarray:
        if self.augmenter is not None:
            frame = self.augmenter(frame, self._t)
        return np.append(self.encoder(preprocess(frame)), self._prev).astype(np.float32)

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        if self.augmenter is not None:
            self.augmenter.new_episode()
        self._t = 0
        self._prev = 0.0
        return self._observe(frame), info

    def step(self, action):
        steer = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        frame, reward, terminated, truncated, info = self.env.step(np.array([steer], dtype=np.float32))
        self._t += 1
        self._prev = steer
        return self._observe(frame), reward, terminated, truncated, info
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wrappers.py -v`
Expected: PASS (5 passed)

- [x] **Step 5: Commit**

```bash
git add envs/wrappers.py tests/test_wrappers.py
git commit -m "feat: add actuator lag and latent observation wrappers"
```

---

### Task 9: Labeled driving data collection

**Files:**
- Create: `collect/drivers.py`, `collect/dataset.py`, `collect/collect.py`
- Test: `tests/test_drivers.py`, `tests/test_dataset.py`, `tests/test_collect.py`

**Interfaces:**
- Consumes: `DonkeyLaneEnv`, `add_env_args`, `env_kwargs_from_args` (Task 5); `SimDisconnectedError` (Task 4); `CAMERA_SHAPE` (Task 2).
- Produces:
  - `ScriptedDriver(rng, cte_sign: float = 1.0, gain: float = 0.5, noise_std: float = 0.2, swerve_prob: float = 0.02, swerve_steps: int = 15)`; `act(cte: float) -> tuple[float, float]` = `(executed_steer, expert_label)`, both in [-1, 1]. The label is `clip(-cte_sign * gain * cte)`; executed = label + noise, or a held hard swerve.
  - `keys_to_steer(left: bool, right: bool, current: float, rate: float = 0.15) -> float` — moves `current` at most `rate` toward `right - left`.
  - `DatasetWriter(data_dir, save_every: int = 200)`: `add(frame, steer, prev_steer, episode: str, source: str)`, `flush()`, properties `total`, attribute `saved`. Writes `frame_<8 digits>.png` and appends to `labels.csv` (`frame, steer, prev_steer, episode, source`), never overwriting.
  - `read_labels(data_dir) -> list[dict]` — keys `frame` (`Path`), `steer` (float), `prev_steer` (float), `episode` (str), `source` (str).
  - `load_frame(path) -> np.ndarray` (RGB uint8; `OSError` if unreadable); `next_frame_index(data_dir) -> int`.
  - `run_scripted(env, driver, writer, n_frames: int, run_id: str)`, `run_manual(env, writer, n_frames: int, run_id: str)`.
  - CLI: `python -m collect.collect --mode {scripted,manual} --out-dir data/drive --n-frames N --cte-sign ±1 --seed S [env flags]`.

Each saved row is the frame the driver *saw* plus the steering decision for it; `prev_steer` is the steering executed on the previous step.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_drivers.py
import numpy as np
import pytest

from collect.drivers import ScriptedDriver, keys_to_steer


def _driver(**kwargs):
    defaults = dict(rng=np.random.default_rng(0), noise_std=0.0, swerve_prob=0.0)
    return ScriptedDriver(**{**defaults, **kwargs})


def test_label_steers_against_cte():
    assert _driver().act(1.0)[1] == pytest.approx(-0.5)
    assert _driver().act(-1.0)[1] == pytest.approx(0.5)
    assert _driver(cte_sign=-1.0).act(1.0)[1] == pytest.approx(0.5)


def test_without_noise_executed_equals_label():
    executed, label = _driver().act(0.4)
    assert executed == label


def test_noise_changes_executed_but_not_label():
    driver = _driver(noise_std=0.3)
    pairs = [driver.act(0.4) for _ in range(20)]
    assert all(label == pytest.approx(-0.2) for _, label in pairs)
    assert any(abs(executed - label) > 0.01 for executed, label in pairs)


def test_outputs_are_clipped():
    executed, label = _driver(gain=10.0).act(5.0)
    assert executed == -1.0 and label == -1.0


def test_swerve_holds_steering_while_label_stays_expert():
    driver = _driver(swerve_prob=1.0, swerve_steps=5)
    pairs = [driver.act(0.0) for _ in range(5)]
    assert len({executed for executed, _ in pairs}) == 1
    assert abs(pairs[0][0]) >= 0.5
    assert all(label == 0.0 for _, label in pairs)


def test_keys_to_steer_ramps_and_returns_to_centre():
    s = 0.0
    for _ in range(3):
        s = keys_to_steer(False, True, s)
    assert s == pytest.approx(0.45)
    for _ in range(20):
        s = keys_to_steer(False, True, s)
    assert s == pytest.approx(1.0)
    s = keys_to_steer(False, False, s)
    assert s == pytest.approx(0.85)
    assert keys_to_steer(True, True, 0.0) == 0.0
```

```python
# tests/test_dataset.py
import numpy as np
import pytest

from car.policy import CAMERA_SHAPE
from collect.dataset import DatasetWriter, load_frame, next_frame_index, read_labels


def _frame(value):
    return np.full(CAMERA_SHAPE, value, dtype=np.uint8)


def test_writer_flushes_in_batches_and_reads_back(tmp_path):
    writer = DatasetWriter(tmp_path, save_every=3)
    for i in range(7):
        writer.add(_frame(i), steer=i / 10, prev_steer=-i / 10, episode="run-0", source="scripted")
    assert len(read_labels(tmp_path)) == 6
    assert writer.total == 7
    writer.flush()
    rows = read_labels(tmp_path)
    assert len(rows) == 7
    assert rows[3]["steer"] == pytest.approx(0.3)
    assert rows[3]["prev_steer"] == pytest.approx(-0.3)
    assert rows[3]["episode"] == "run-0" and rows[3]["source"] == "scripted"
    assert np.array_equal(load_frame(rows[3]["frame"]), _frame(3))


def test_second_writer_appends(tmp_path):
    for value in (1, 2):
        writer = DatasetWriter(tmp_path)
        writer.add(_frame(value), 0.0, 0.0, f"run-{value}", "manual")
        writer.flush()
    rows = read_labels(tmp_path)
    assert [r["episode"] for r in rows] == ["run-1", "run-2"]
    assert np.array_equal(load_frame(rows[1]["frame"]), _frame(2))


def test_deleting_a_frame_never_causes_overwrite(tmp_path):
    writer = DatasetWriter(tmp_path)
    for _ in range(3):
        writer.add(_frame(0), 0.0, 0.0, "e", "scripted")
    writer.flush()
    (tmp_path / "frame_00000000.png").unlink()
    assert next_frame_index(tmp_path) == 3


def test_rgb_order_survives_roundtrip(tmp_path):
    frame = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    frame[..., 0] = 200
    writer = DatasetWriter(tmp_path)
    writer.add(frame, 0.0, 0.0, "e", "scripted")
    writer.flush()
    assert np.array_equal(load_frame(read_labels(tmp_path)[0]["frame"]), frame)


def test_read_labels_of_missing_dir_is_empty(tmp_path):
    assert read_labels(tmp_path / "nothing") == []
```

```python
# tests/test_collect.py
import numpy as np
import pytest

from car.policy import CAMERA_SHAPE
from collect.collect import run_scripted
from collect.dataset import DatasetWriter, load_frame, read_labels
from collect.drivers import ScriptedDriver


class FakeEnv:
    """cte is 0 after reset and 1.0 after every step; crashes every 4 steps."""

    def __init__(self):
        self.resets = 0
        self.steps = 0
        self.actions = []

    def reset(self):
        self.resets += 1
        return np.zeros(CAMERA_SHAPE, dtype=np.uint8), {"cte": 0.0}

    def step(self, action):
        self.steps += 1
        self.actions.append(float(action[0]))
        terminated = self.steps % 4 == 0
        return np.full(CAMERA_SHAPE, self.steps, dtype=np.uint8), 0.0, terminated, False, {"cte": 1.0}


def test_run_scripted_aligns_frames_labels_and_previous_steering(tmp_path):
    env = FakeEnv()
    writer = DatasetWriter(tmp_path, save_every=5)
    driver = ScriptedDriver(np.random.default_rng(0), noise_std=0.0, swerve_prob=0.0)
    run_scripted(env, driver, writer, n_frames=10, run_id="run")
    writer.flush()
    rows = read_labels(tmp_path)
    assert len(rows) == 10
    assert env.resets == 3  # initial + crashes at steps 4 and 8
    # row 0: reset frame, cte 0
    assert rows[0]["steer"] == 0.0 and rows[0]["prev_steer"] == 0.0
    # row 1: frame after step 1 (pixel value 1), cte 1 -> label -0.5, previous executed steer 0
    assert np.array_equal(load_frame(rows[1]["frame"]), np.full(CAMERA_SHAPE, 1, dtype=np.uint8))
    assert rows[1]["steer"] == pytest.approx(-0.5) and rows[1]["prev_steer"] == 0.0
    assert rows[2]["prev_steer"] == pytest.approx(-0.5)
    # step 4 crashed: row 4 starts a new episode with prev_steer reset
    assert rows[3]["episode"] == "run-0" and rows[4]["episode"] == "run-1"
    assert rows[4]["prev_steer"] == 0.0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_drivers.py tests/test_dataset.py tests/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write `collect/drivers.py`**

```python
# collect/drivers.py
"""Non-learned drivers used to record training data. Steering in [-1, 1]."""
import numpy as np


class ScriptedDriver:
    """Proportional lane-follower on CTE that deliberately drives imperfectly.

    act() returns (executed, label):
      label    = what a clean expert would steer here: clip(-cte_sign * gain * cte)
      executed = label + Gaussian noise, or a held hard swerve now and then
    The car follows the noisy executed steering, so the frames show
    off-centre and recovering views; BC learns the clean label for each of
    them, i.e. how to get back to the centre (noise injection with expert
    labels, as in DART).
    cte_sign flips the controller if the sim's CTE sign is the opposite of
    what we assume; calibrate it with scripts/check_sim.py.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        cte_sign: float = 1.0,
        gain: float = 0.5,
        noise_std: float = 0.2,
        swerve_prob: float = 0.02,
        swerve_steps: int = 15,
    ):
        self.rng = rng
        self.cte_sign = cte_sign
        self.gain = gain
        self.noise_std = noise_std
        self.swerve_prob = swerve_prob
        self.swerve_steps = swerve_steps
        self._swerve_left = 0
        self._swerve_steer = 0.0

    def act(self, cte: float) -> tuple[float, float]:
        label = float(np.clip(-self.cte_sign * self.gain * cte, -1.0, 1.0))
        if self._swerve_left == 0 and self.rng.random() < self.swerve_prob:
            self._swerve_left = self.swerve_steps
            self._swerve_steer = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.5, 1.0))
        if self._swerve_left > 0:
            self._swerve_left -= 1
            return self._swerve_steer, label
        executed = float(np.clip(label + self.rng.normal(0.0, self.noise_std), -1.0, 1.0))
        return executed, label


def keys_to_steer(left: bool, right: bool, current: float, rate: float = 0.15) -> float:
    """Arrow keys -> steering that ramps toward the pressed direction (and back
    to 0 when released), so human labels aren't only -1 / 0 / +1."""
    target = float(right) - float(left)
    return float(np.clip(current + np.clip(target - current, -rate, rate), -1.0, 1.0))
```

- [x] **Step 4: Write `collect/dataset.py`**

```python
# collect/dataset.py
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
```

- [x] **Step 5: Write `collect/collect.py`**

```python
# collect/collect.py
"""Drive the simulator and record frames with steering labels.

  --mode scripted   noisy lane-follower labelled with clean expert steering (unattended)
  --mode manual     you drive with the arrow keys in a small pygame window; Esc stops

Runs append to --out-dir; re-running adds more data. Use the same env flags
(--env-name, --throttle, --cte-max, --cam-fov) as for training.
"""
import argparse
import time

import numpy as np

from collect.dataset import DatasetWriter
from collect.drivers import ScriptedDriver, keys_to_steer
from envs.cli import add_env_args, env_kwargs_from_args
from envs.donkey_env import DonkeyLaneEnv
from envs.sim_process import SimDisconnectedError


def run_scripted(env, driver: ScriptedDriver, writer: DatasetWriter, n_frames: int, run_id: str) -> None:
    frame, info = env.reset()
    episode, prev = 0, 0.0
    while writer.total < n_frames:
        executed, label = driver.act(info.get("cte", 0.0))
        writer.add(frame, label, prev, f"{run_id}-{episode}", "scripted")
        frame, _, terminated, truncated, info = env.step(np.array([executed], dtype=np.float32))
        prev = executed
        if terminated or truncated:
            frame, info = env.reset()
            episode, prev = episode + 1, 0.0


def run_manual(env, writer: DatasetWriter, n_frames: int, run_id: str) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((480, 360))
    pygame.display.set_caption("Donkey manual drive: left/right arrows steer, Esc stops")
    frame, _ = env.reset()
    episode, prev, steer = 0, 0.0, 0.0
    running = True
    while running and writer.total < n_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        keys = pygame.key.get_pressed()
        steer = keys_to_steer(keys[pygame.K_LEFT], keys[pygame.K_RIGHT], steer)
        writer.add(frame, steer, prev, f"{run_id}-{episode}", "manual")
        frame, _, terminated, truncated, _ = env.step(np.array([steer], dtype=np.float32))
        prev = steer
        surface = pygame.surfarray.make_surface(np.ascontiguousarray(frame.swapaxes(0, 1)))
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.flip()
        if terminated or truncated:
            frame, _ = env.reset()
            episode, prev, steer = episode + 1, 0.0, 0.0
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(parser)
    parser.add_argument("--mode", choices=["scripted", "manual"], default="scripted")
    parser.add_argument("--out-dir", default="data/drive")
    parser.add_argument("--n-frames", type=int, default=3000, help="frames to record in this run")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--cte-sign", type=float, default=1.0, help="scripted mode: +1 or -1, see check_sim")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_id = f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    writer = DatasetWriter(args.out_dir, args.save_every)
    env = DonkeyLaneEnv(**env_kwargs_from_args(args))
    try:
        if args.mode == "scripted":
            driver = ScriptedDriver(np.random.default_rng(args.seed), cte_sign=args.cte_sign)
            run_scripted(env, driver, writer, args.n_frames, run_id)
        else:
            run_manual(env, writer, args.n_frames, run_id)
    except SimDisconnectedError as exc:
        print(f"simulator disconnected: {exc}. Re-run the command to keep appending.")
    finally:
        writer.flush()
        env.close()
    print(f"done: {writer.saved} frames written to {args.out_dir} this run")


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_drivers.py tests/test_dataset.py tests/test_collect.py -v`
Expected: PASS (12 passed)

- [x] **Step 7: Manual — both modes work on the real sim**

Use the calibrated flags from Task 6 (written below as `<flags>` = `--env-name ... --throttle ... --cte-max ...`).
- `python -m collect.collect --mode scripted --out-dir data/drive_check --n-frames 400 --cte-sign <sign> <flags>` — the car follows the lane with visible wobble and occasional swerves; ends with `done: 400 frames written`. If the car steers straight off the road every episode, flip `--cte-sign`.
- `python -m collect.collect --mode manual --out-dir data/drive_check --n-frames 200 <flags>` — click the pygame window; left/right arrows steer; Esc stops.
- `head -5 data/drive_check/labels.csv` shows rows; then `rm -r data/drive_check`.

- [x] **Step 8: Commit**

```bash
git add collect tests/test_drivers.py tests/test_dataset.py tests/test_collect.py
git commit -m "feat: add labeled driving data collection"
```

---

### Task 10: VAE training (clean and denoising-DR)

**Files:**
- Create: `learn/train_vae.py`
- Test: `tests/test_train_vae.py`

**Interfaces:**
- Consumes: `preprocess` (Task 2); `ConvVAE`, `save_vae`, `encoder_arrays`, `save_npz` (Task 3); `DomainRandomizer` (Task 7); `read_labels`, `load_frame` (Task 9).
- Produces:
  - `DriveFrames(paths, dr: bool, seed: int = 0)` — items `(input, target)`, both `(3,80,80)` float32 tensors; `target` is the clean frame, `input` equals it unless `dr`, then it is a freshly randomized copy.
  - `seed_worker(worker_id)` — DataLoader `worker_init_fn` giving each worker its own augmentation RNG.
  - `vae_loss(recon, target, mu, logvar, beta: float = 1.0) -> torch.Tensor`
  - `run_epoch(model, loader, device, optimizer=None, beta=1.0) -> float`
  - `train(model, train_loader, val_loader, epochs: int, device: str = "cpu", lr: float = 1e-3, beta: float = 1.0) -> dict` (`{"train": [...], "val": [...]}`; model left holding best-validation weights)
  - `save_reconstructions(model, dataset, out_path, n: int = 8)` — PNG rows: input, target, reconstruction.
  - CLI: `python -m learn.train_vae --data-dir data/drive --out-dir models/vae_clean [--dr]` → `vae.pth`, `encoder.npz`, `recon.png`, `history.json` in `--out-dir`. Refuses fewer than 500 frames.

- [x] **Step 1: Write the failing test**

```python
# tests/test_train_vae.py
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_vae.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'learn.train_vae'`

- [x] **Step 3: Write the implementation**

```python
# learn/train_vae.py
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
    parser.add_argument("--data-dir", default="data/drive")
    parser.add_argument("--out-dir", default="models/vae_clean")
    parser.add_argument("--dr", action="store_true", help="denoising VAE on domain-randomized inputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = [row["frame"] for row in read_labels(args.data_dir)]
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
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_vae.py -v`
Expected: PASS (5 passed)

- [x] **Step 5: Commit**

```bash
git add learn/train_vae.py tests/test_train_vae.py
git commit -m "feat: add clean and denoising-DR VAE training"
```

---

### Task 11: Behavioral cloning head training

**Files:**
- Create: `learn/train_bc.py`
- Test: `tests/test_train_bc.py`

**Interfaces:**
- Consumes: `preprocess`, `Policy` (Task 2); `PolicyHead`, `load_vae`, `encoder_arrays`, `head_arrays`, `save_npz` (Task 3); `DomainRandomizer` (Task 7); `read_labels`, `load_frame` (Task 9); `seed_worker` (Task 10).
- Produces:
  - `BCData(rows, dr: bool, seed: int = 0, prev_noise: float = 0.0)` — items `(image (3,80,80), prev_steer (1,), steer (1,))` float32 tensors.
  - `split_by_episode(rows, val_fraction: float, seed: int) -> tuple[list, list]` — `ValueError` with fewer than 2 episodes.
  - `run_epoch(encoder, head, loader, device, optimizer=None) -> float` (mean MSE per batch; encoder never trained).
  - `train(encoder, head, train_loader, val_loader, epochs: int, device: str = "cpu", lr: float = 1e-3) -> dict` (head left holding best-validation weights).
  - CLI: `python -m learn.train_bc --vae models/vae_clean/vae.pth --out models/bc_clean/policy.npz [--dr] [--data-dir data/drive]` → `policy.npz` (encoder + head) and `history.json` next to it.

- [x] **Step 1: Write the failing test**

```python
# tests/test_train_bc.py
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_bc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'learn.train_bc'`

- [x] **Step 3: Write the implementation**

```python
# learn/train_bc.py
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
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_bc.py -v`
Expected: PASS (4 passed)

- [x] **Step 5: Commit**

```bash
git add learn/train_bc.py tests/test_train_bc.py
git commit -m "feat: add behavioral cloning head training"
```

---

### Task 12: Crash-resumable SAC training with export

**Files:**
- Create: `learn/sac_config.py`, `learn/train_sac.py`
- Test: `tests/test_train_sac.py`

**Interfaces:**
- Consumes: `Encoder`, `Head`, `load_arrays` (Task 2); `PolicyHead`, `head_arrays`, `encoder_arrays`, `save_npz` (Task 3); `SimDisconnectedError` (Task 4); `DonkeyLaneEnv`, `add_env_args`, `env_kwargs_from_args` (Task 5); `DomainRandomizer` (Task 7); `LatentEnv` (Task 8).
- Produces:
  - `SAC_CONFIG: dict` — kwargs for `stable_baselines3.SAC` (all except `env`, `verbose`, `tensorboard_log`).
  - `latest_checkpoint(checkpoint_dir) -> tuple[Path, Path | None] | None` (SB3 `CheckpointCallback(name_prefix="sac", save_replay_buffer=True)` names: `sac_<steps>_steps.zip`, `sac_replay_buffer_<steps>_steps.pkl`).
  - `train(make_env, total_timesteps: int, run_dir, checkpoint_freq: int = 5000, max_restarts: int = 5, config: dict = SAC_CONFIG, tensorboard: bool = True) -> SAC`
  - `export_sac_policy(model: SAC, encoder: dict, out_path) -> None` — writes encoder arrays + the actor copied into `PolicyHead`; `car.policy.Head` then equals `model.predict(obs, deterministic=True)`.
  - CLI: `python -m learn.train_sac --encoder models/vae_clean/encoder.npz --run-dir models/sac_clean [--dr] [--total-timesteps 80000] [env flags]` → `<run-dir>/checkpoints/`, `tb/`, `settings.json`, `sac_final.zip`, `policy.npz`. Refuses to reuse a run dir whose saved settings differ.

- [x] **Step 1: Write `learn/sac_config.py`**

```python
# learn/sac_config.py
"""SAC hyperparameters: a starting point modelled on published Donkey-sim +
VAE setups. net_arch must stay two equal hidden layers: the exported head
(learn/nets.py::PolicyHead, car/policy.py::Head) has exactly that shape,
and BC uses the same size."""

SAC_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 7.3e-4,
    "buffer_size": 100_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.02,
    "ent_coef": "auto",
    # The sim runs in real time: gradient updates between env steps would
    # delay every command, so collect a whole episode, then do one gradient
    # step per collected step (-1) while the car is being reset.
    "train_freq": (1, "episode"),
    "gradient_steps": -1,
    "learning_starts": 1000,
    # State-dependent exploration: smoother steering than per-step noise.
    # Also what makes the actor's mean clip to [-2, 2] (mirrored in the head).
    "use_sde": True,
    "sde_sample_freq": 16,
    "policy_kwargs": {"log_std_init": -2, "net_arch": [64, 64]},
}
```

- [x] **Step 2: Write the failing test**

`FakeLatentEnv` subclasses `gymnasium.Env` because SB3 wraps envs in `Monitor`, which requires it.

```python
# tests/test_train_sac.py
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import SAC

from car.policy import Head, load_arrays
from envs.sim_process import SimDisconnectedError
from learn.nets import Encoder, encoder_arrays
from learn.sac_config import SAC_CONFIG
from learn.train_sac import export_sac_policy, latest_checkpoint, train

TINY_CONFIG = {
    **SAC_CONFIG,
    "buffer_size": 1000,
    "batch_size": 16,
    "learning_starts": 10,
    "policy_kwargs": {"log_std_init": -2, "net_arch": [16, 16]},
}


class FakeLatentEnv(gym.Env):
    """33-dim observations, 5-step episodes; optionally raises
    SimDisconnectedError on a given total step to simulate the sim dying."""

    def __init__(self, crash_at_step=None):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.crash_at_step = crash_at_step
        self.total_steps = 0
        self.episode_steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        return np.zeros(33, dtype=np.float32), {}

    def step(self, action):
        self.total_steps += 1
        self.episode_steps += 1
        if self.total_steps == self.crash_at_step:
            raise SimDisconnectedError("fake crash")
        obs = np.random.standard_normal(33).astype(np.float32)
        return obs, float(-abs(action[0])), False, self.episode_steps >= 5, {}

    def close(self):
        self.closed = True


def test_latest_checkpoint_picks_highest_steps(tmp_path):
    for steps in (10, 200, 30):
        (tmp_path / f"sac_{steps}_steps.zip").touch()
    (tmp_path / "sac_replay_buffer_200_steps.pkl").touch()
    model_path, buffer_path = latest_checkpoint(tmp_path)
    assert model_path.name == "sac_200_steps.zip"
    assert buffer_path.name == "sac_replay_buffer_200_steps.pkl"


def test_latest_checkpoint_without_buffer_or_files(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    (tmp_path / "sac_50_steps.zip").touch()
    assert latest_checkpoint(tmp_path)[1] is None


def test_train_runs_to_total_timesteps(tmp_path):
    envs = []

    def make_env():
        envs.append(FakeLatentEnv())
        return envs[-1]

    model = train(make_env, total_timesteps=40, run_dir=tmp_path, checkpoint_freq=10, config=TINY_CONFIG, tensorboard=False)
    assert model.num_timesteps >= 40
    assert latest_checkpoint(tmp_path / "checkpoints") is not None
    assert envs[-1].closed


def test_train_resumes_from_checkpoint_after_sim_crash(tmp_path):
    envs = []

    def make_env():
        envs.append(FakeLatentEnv(crash_at_step=33 if not envs else None))
        return envs[-1]

    model = train(make_env, total_timesteps=60, run_dir=tmp_path, checkpoint_freq=10, config=TINY_CONFIG, tensorboard=False)
    assert len(envs) == 2
    assert envs[0].closed
    assert model.num_timesteps >= 60
    assert envs[1].total_steps <= 35  # resumed at the 30-step checkpoint


def test_exported_head_matches_sac_predict(tmp_path):
    model = SAC(env=FakeLatentEnv(), verbose=0, **TINY_CONFIG)
    model.learn(40)
    path = tmp_path / "policy.npz"
    export_sac_policy(model, encoder_arrays(Encoder()), path)
    arrays = load_arrays(path)
    assert any(k.startswith("encoder.") for k in arrays)
    head = Head(arrays)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = (rng.standard_normal(33) * 3).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        assert head(obs) == pytest.approx(float(action[0]), abs=1e-5)
```

- [x] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_train_sac.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'learn.train_sac'`

- [x] **Step 4: Write `learn/train_sac.py`**

```python
# learn/train_sac.py
"""Train SAC on the latent Donkey env, surviving simulator crashes, then export
the actor as policy.npz (same format as BC).

Everything for one run lives in --run-dir:
  checkpoints/    model + replay buffer every --checkpoint-freq steps
  tb/             TensorBoard logs (tensorboard --logdir <run-dir>/tb)
  settings.json   env flags, encoder, dr, seed; a rerun with different flags is refused
  sac_final.zip   final SB3 model
  policy.npz      encoder + exported head, for the benchmark and the car

If the sim dies, the env is rebuilt and training resumes from the latest
checkpoint (at most --max-restarts times). Re-running the same command after
Ctrl+C or a reboot also resumes.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from car.policy import Encoder, load_arrays
from envs.augment import DomainRandomizer
from envs.cli import add_env_args, env_kwargs_from_args
from envs.donkey_env import DonkeyLaneEnv
from envs.sim_process import SimDisconnectedError
from envs.wrappers import LatentEnv
from learn.nets import PolicyHead, head_arrays, save_npz
from learn.sac_config import SAC_CONFIG

_CHECKPOINT_NAME = re.compile(r"sac_(\d+)_steps\.zip$")


def latest_checkpoint(checkpoint_dir):
    """(model zip, replay buffer pkl or None) for the highest step count, or None."""
    checkpoint_dir = Path(checkpoint_dir)
    found = [(int(m.group(1)), p) for p in checkpoint_dir.glob("sac_*_steps.zip") if (m := _CHECKPOINT_NAME.search(p.name))]
    if not found:
        return None
    steps, model_path = max(found)
    buffer_path = checkpoint_dir / f"sac_replay_buffer_{steps}_steps.pkl"
    return model_path, buffer_path if buffer_path.exists() else None


def train(
    make_env,
    total_timesteps: int,
    run_dir,
    checkpoint_freq: int = 5000,
    max_restarts: int = 5,
    config: dict = SAC_CONFIG,
    tensorboard: bool = True,
) -> SAC:
    """Train until total_timesteps, rebuilding the env and resuming from the
    latest checkpoint whenever it raises SimDisconnectedError."""
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    restarts = 0
    while True:
        env = None
        try:
            env = make_env()
            found = latest_checkpoint(checkpoint_dir)
            if found:
                model_path, buffer_path = found
                print(f"resuming from {model_path}")
                model = SAC.load(model_path, env=env)
                if buffer_path is not None:
                    model.load_replay_buffer(buffer_path)
            else:
                model = SAC(env=env, verbose=1, tensorboard_log=str(run_dir / "tb") if tensorboard else None, **config)
            remaining = total_timesteps - model.num_timesteps
            if remaining > 0:
                callback = CheckpointCallback(
                    save_freq=checkpoint_freq, save_path=str(checkpoint_dir), name_prefix="sac", save_replay_buffer=True
                )
                model.learn(remaining, callback=callback, reset_num_timesteps=False, tb_log_name="sac")
            return model
        except SimDisconnectedError as exc:
            restarts += 1
            if restarts > max_restarts:
                raise
            print(f"simulator disconnected ({exc}); restart {restarts}/{max_restarts} from latest checkpoint")
        finally:
            if env is not None:
                env.close()


def export_sac_policy(model: SAC, encoder: dict, out_path) -> None:
    """Copy SAC's deterministic actor (latent_pi MLP + mean layer; gSDE wraps the
    mean layer with a [-2, 2] Hardtanh) into PolicyHead and save it with the encoder."""
    actor = model.actor
    first = actor.latent_pi[0]
    head = PolicyHead(obs_dim=first.in_features, hidden=first.out_features)
    head.latent_pi.load_state_dict(actor.latent_pi.state_dict())
    mu = actor.mu[0] if isinstance(actor.mu, nn.Sequential) else actor.mu
    head.mu.load_state_dict(mu.state_dict())
    save_npz(out_path, {**encoder, **head_arrays(head)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(parser)
    parser.add_argument("--encoder", required=True, help="encoder.npz from learn.train_vae")
    parser.add_argument("--run-dir", required=True, help="e.g. models/sac_clean")
    parser.add_argument("--dr", action="store_true", help="new domain randomization every episode")
    parser.add_argument("--total-timesteps", type=int, default=80_000)
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_kwargs = env_kwargs_from_args(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        **{k: v for k, v in env_kwargs.items() if k not in ("exe_path", "port")},
        "encoder": args.encoder,
        "dr": args.dr,
        "seed": args.seed,
    }
    settings_path = run_dir / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise SystemExit(f"{settings_path} differs from these flags; use a new --run-dir for new settings")
    settings_path.write_text(json.dumps(settings, indent=2))

    encoder_weights = load_arrays(args.encoder)
    encoder = Encoder(encoder_weights)
    rng = np.random.default_rng(args.seed)

    def make_env():
        augmenter = DomainRandomizer(rng) if args.dr else None
        return LatentEnv(DonkeyLaneEnv(**env_kwargs), encoder, augmenter)

    model = train(make_env, args.total_timesteps, run_dir, args.checkpoint_freq, args.max_restarts)
    model.save(run_dir / "sac_final")
    export_sac_policy(model, encoder_weights, run_dir / "policy.npz")
    print(f"saved {run_dir / 'sac_final.zip'} and {run_dir / 'policy.npz'}")


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train_sac.py -v`
Expected: PASS (5 passed), under a minute on CPU.
If `test_exported_head_matches_sac_predict` fails, print `model.actor` to see how this SB3 version builds the mean layer and adjust `export_sac_policy` (and, if SB3 no longer clips the gSDE mean, `CLIP_MEAN` handling in `car/policy.py` and `learn/nets.py`) until it passes. Do not loosen the tolerance.

- [x] **Step 6: Commit**

```bash
git add learn/sac_config.py learn/train_sac.py tests/test_train_sac.py
git commit -m "feat: add crash-resumable SAC training with policy export"
```

---

### Task 13: Conditions and metrics

**Files:**
- Create: `bench/conditions.py`, `bench/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `AUGMENTERS` (Task 7, test only).
- Produces:
  - `MODELS = ("sac_clean", "sac_dr", "bc_clean", "bc_dr")`
  - `Condition(name, augmenter=None, steer_delay=0, throttle_scale=1.0, offset_start=False)` (frozen dataclass); `CONDITIONS` (tuple, spec order); `BY_NAME: dict[str, Condition]`; `NOISE_CONDITIONS` (names except `nominal`).
  - `steering_jerk(steer, t) -> float` (mean |Δsteer/Δt|; `nan` with < 2 samples)
  - `oscillation_counts(steer, t, window=21, deadband=0.02) -> tuple[int, float]` (sign changes, duration)
  - `oscillation_hz(steer, t, window=21, deadband=0.02) -> float` (`nan` if too short)
  - `wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]`
  - `parse_trial_log(path) -> dict` — keys `takeover_times` (list[float]), `pilot_start`, `jerk`, `osc_hz`, `latency_ms_mean`, `latency_ms_p95` (floats, `nan` when absent). Reads the car log columns `time, mode, pilot_steer, user_steer, latency_ms`; any mode other than `"user"` counts as pilot.
  - `trial_outcome(log: dict, completed: bool) -> dict` — keys `completed`, `interventions` (int), `clean_lap` (bool), `lap_time` (float or `nan`), `early_intervention` (bool; an intervention within 5 s of the first pilot sample).

A completed lap ends with one final takeover at the finish line; every other takeover is an intervention.

- [x] **Step 1: Write `bench/conditions.py`**

```python
# bench/conditions.py
"""The six evaluation conditions (same names in sim and on the real car) and the four models."""
from dataclasses import dataclass

MODELS = ("sac_clean", "sac_dr", "bc_clean", "bc_dr")


@dataclass(frozen=True)
class Condition:
    name: str
    augmenter: str | None = None  # envs.augment name applied to camera frames (sim)
    steer_delay: int = 0  # steps of steering delay (sim)
    throttle_scale: float = 1.0  # multiplier on the fixed sim throttle
    offset_start: bool = False  # push the car off its line before handing over


CONDITIONS = (
    Condition("nominal"),
    Condition("low_light", augmenter="low_light"),
    Condition("shadows", augmenter="shadows"),
    Condition("distractors", augmenter="distractors"),
    Condition("low_battery", steer_delay=3, throttle_scale=0.7),
    Condition("offset_start", offset_start=True),
)
BY_NAME = {c.name: c for c in CONDITIONS}
NOISE_CONDITIONS = tuple(c.name for c in CONDITIONS if c.name != "nominal")
```

- [x] **Step 2: Write the failing test**

```python
# tests/test_metrics.py
import csv
import math

import numpy as np
import pytest

from bench.conditions import CONDITIONS, NOISE_CONDITIONS
from bench.metrics import oscillation_hz, parse_trial_log, steering_jerk, trial_outcome, wilson_interval
from envs.augment import AUGMENTERS

LOG_FIELDS = ("time", "mode", "pilot_steer", "user_steer", "latency_ms")


def write_log(path, modes, steer=None, dt=0.05, latency=12.0):
    steer = steer if steer is not None else [0.0] * len(modes)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_FIELDS)
        for i, (mode, s) in enumerate(zip(modes, steer)):
            writer.writerow([f"{1000 + i * dt:.3f}", mode, f"{s:.4f}", "0.0000", f"{latency:.2f}"])


def test_condition_augmenters_exist():
    assert all(c.augmenter is None or c.augmenter in AUGMENTERS for c in CONDITIONS)
    assert "nominal" not in NOISE_CONDITIONS and len(NOISE_CONDITIONS) == 5


def test_steering_jerk_of_ramp():
    t = np.arange(0, 1, 0.05)
    assert steering_jerk(0.1 * t, t) == pytest.approx(0.1)
    assert math.isnan(steering_jerk([0.2], [0.0]))


def test_oscillation_of_two_hertz_sine():
    t = np.arange(0, 10, 0.05)
    assert oscillation_hz(0.3 * np.sin(2 * np.pi * 2.0 * t), t) == pytest.approx(2.0, abs=0.2)


def test_oscillation_of_constant_and_short_signals():
    t = np.arange(0, 5, 0.05)
    assert oscillation_hz(np.full(len(t), 0.4), t) == 0.0
    assert math.isnan(oscillation_hz([0.1, 0.2], [0.0, 0.05]))


def test_wilson_interval():
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    lo, hi = wilson_interval(5, 10)
    assert lo + hi == pytest.approx(1.0)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_parse_trial_log(tmp_path):
    # 10 user, 100 pilot, 10 user (intervention), 60 pilot, 5 user (finish)
    modes = ["user"] * 10 + ["local"] * 100 + ["user"] * 10 + ["local"] * 60 + ["user"] * 5
    steer = [0.3 * math.sin(2 * math.pi * 1.0 * i * 0.05) for i in range(len(modes))]
    path = tmp_path / "trial_001.csv"
    write_log(path, modes, steer)
    log = parse_trial_log(path)
    assert log["takeover_times"] == pytest.approx([1005.5, 1009.0])
    assert log["pilot_start"] == pytest.approx(1000.5)
    assert log["latency_ms_mean"] == pytest.approx(12.0)
    assert log["osc_hz"] == pytest.approx(1.0, abs=0.25)
    assert log["jerk"] > 0


def test_parse_log_without_pilot_samples(tmp_path):
    path = tmp_path / "t.csv"
    write_log(path, ["user"] * 5)
    log = parse_trial_log(path)
    assert log["takeover_times"] == [] and math.isnan(log["jerk"])


def test_trial_outcome():
    log = {"takeover_times": [1005.5, 1009.0], "pilot_start": 1000.5}
    done = trial_outcome(log, completed=True)
    assert done["interventions"] == 1 and not done["clean_lap"]
    assert math.isnan(done["lap_time"])
    assert done["early_intervention"]  # 5.0 s after start
    clean = trial_outcome({"takeover_times": [1009.0], "pilot_start": 1000.5}, completed=True)
    assert clean["clean_lap"] and clean["lap_time"] == pytest.approx(8.5)
    assert not clean["early_intervention"]
    failed = trial_outcome(log, completed=False)
    assert failed["interventions"] == 2 and not failed["clean_lap"]
```

- [x] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.metrics'`

- [x] **Step 4: Write `bench/metrics.py`**

```python
# bench/metrics.py
"""Metrics computed the same way for simulator episodes and real-car trials."""
import csv
import math

import numpy as np

OSC_WINDOW = 21  # samples in the moving average (about 1 s at 20 Hz)
OSC_DEADBAND = 0.02  # steering wiggles smaller than this are ignored
EARLY_SECONDS = 5.0


def steering_jerk(steer, t) -> float:
    """Mean |d steering / dt| in 1/s."""
    s, t = np.asarray(steer, dtype=float), np.asarray(t, dtype=float)
    if len(s) < 2:
        return math.nan
    dt = np.diff(t)
    ok = dt > 0
    return float(np.mean(np.abs(np.diff(s)[ok] / dt[ok]))) if ok.any() else math.nan


def oscillation_counts(steer, t, window: int = OSC_WINDOW, deadband: float = OSC_DEADBAND) -> tuple[int, float]:
    """(sign changes of steering minus its moving average, duration they span)."""
    s, t = np.asarray(steer, dtype=float), np.asarray(t, dtype=float)
    if len(s) < window:
        return 0, 0.0
    half = window // 2
    detrended = (s - np.convolve(s, np.ones(window) / window, mode="same"))[half : len(s) - half]
    signs = np.sign(detrended[np.abs(detrended) >= deadband])
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return changes, float(t[len(t) - 1 - half] - t[half])


def oscillation_hz(steer, t, window: int = OSC_WINDOW, deadband: float = OSC_DEADBAND) -> float:
    """Weaving frequency: two sign changes around the moving average per cycle."""
    changes, duration = oscillation_counts(steer, t, window, deadband)
    return changes / 2 / duration if duration > 0 else math.nan


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate; sensible for small n."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _runs(mask) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    runs, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def parse_trial_log(path) -> dict:
    """Summarise one real-car drive log written by car/pilot_part.py."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time"]) for r in rows])
    pilot = np.array([r["mode"] != "user" for r in rows], dtype=bool)
    steer = np.array([float(r["pilot_steer"]) for r in rows])
    latency = np.array([float(r["latency_ms"]) for r in rows])
    if not pilot.any():
        return {"takeover_times": [], "pilot_start": math.nan, "jerk": math.nan, "osc_hz": math.nan,
                "latency_ms_mean": math.nan, "latency_ms_p95": math.nan}

    takeover_times = [float(t[i]) for i in range(1, len(rows)) if pilot[i - 1] and not pilot[i]]
    rates, changes, duration = [], 0, 0.0
    for a, b in _runs(pilot):  # metrics only inside autopilot stretches
        seg_t, seg_s = t[a:b], steer[a:b]
        dt = np.diff(seg_t)
        ok = dt > 0
        rates.extend(np.abs(np.diff(seg_s)[ok] / dt[ok]))
        c, d = oscillation_counts(seg_s, seg_t)
        changes, duration = changes + c, duration + d
    return {
        "takeover_times": takeover_times,
        "pilot_start": float(t[np.argmax(pilot)]),
        "jerk": float(np.mean(rates)) if rates else math.nan,
        "osc_hz": changes / 2 / duration if duration > 0 else math.nan,
        "latency_ms_mean": float(latency[pilot].mean()),
        "latency_ms_p95": float(np.percentile(latency[pilot], 95)),
    }


def trial_outcome(log: dict, completed: bool) -> dict:
    """A completed lap ends with the operator's takeover at the finish line;
    every other takeover was an intervention."""
    times = log["takeover_times"]
    interventions = times[:-1] if completed and times else times
    clean = bool(completed and not interventions)
    return {
        "completed": bool(completed),
        "interventions": len(interventions),
        "clean_lap": clean,
        "lap_time": times[-1] - log["pilot_start"] if clean and times else math.nan,
        "early_intervention": any(x - log["pilot_start"] <= EARLY_SECONDS for x in interventions),
    }
```

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (8 passed)

- [x] **Step 6: Commit**

```bash
git add bench/conditions.py bench/metrics.py tests/test_metrics.py
git commit -m "feat: add evaluation conditions and shared metrics"
```

---

### Task 14: Resumable simulator benchmark

**Files:**
- Create: `bench/sim_bench.py`
- Test: `tests/test_sim_bench.py`

**Interfaces:**
- Consumes: `Policy` (Task 2); `SimDisconnectedError` (Task 4); `DonkeyLaneEnv`, `add_env_args`, `env_kwargs_from_args` (Task 5); `make_augmenter` (Task 7); `ActuatorLag`, `LatentEnv` (Task 8); `Condition`, `CONDITIONS`, `BY_NAME`, `MODELS` (Task 13); `steering_jerk`, `oscillation_hz` (Task 13).
- Produces:
  - `run_episode(head, env, condition, cte_max: float, direction: float = 1.0, laps: int = 1, max_offset_steps: int = 30, clock=time.monotonic) -> dict` — keys `outcome` (`success`|`crash`|`timeout`), `steps`, `avg_abs_cte`, `max_abs_cte`, `jerk`, `osc_hz`, `lap_time` (float or `None`), `recovery_steps` (int or `None`). With `offset_start`, steering is forced to `0.6 * direction` until `|cte| ≥ 0.5 * cte_max` or `max_offset_steps`; forced steps are not passed to `head` nor counted in jerk/oscillation. `recovery_steps` = steps after handover until `|cte| < 0.2 * cte_max`.
  - `plan_runs(models, conditions, n_episodes) -> list[tuple[str, Condition, int]]` — order: condition, then episode index, then model.
  - `run_bench(env_factory, policies: dict, conditions, n_episodes: int, out_path, base_throttle: float, cte_max: float, laps: int = 1, max_restarts: int = 5, clock=time.monotonic) -> int` — `env_factory() -> (raw, lag, top)` where `raw.throttle`, `lag.delay_steps`, `top.encoder`, `top.augmenter` are set before each episode and episodes run on `top`. `policies[name]` has `.encoder` and `.head`. Appends one JSON line per episode (`model`, `condition`, `episode` + `run_episode` keys) and skips episodes already in the file. Augmenter RNG seeded from `(crc32(condition name), episode)`, so every model sees the same perturbation. Rebuilds the env on `SimDisconnectedError` (at most `max_restarts`). Returns the number of episodes run.
  - CLI: `python -m bench.sim_bench [--models name=path ...] [--conditions ...] [--n-episodes 10] [--laps 1] [--out results/sim/episodes.jsonl] [env flags]`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_sim_bench.py
import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.conditions import BY_NAME
from bench.sim_bench import plan_runs, run_bench, run_episode
from envs.sim_process import SimDisconnectedError


class ScriptedEnv:
    """Replays (info, terminated, truncated) per step; records actions."""

    def __init__(self, steps):
        self.steps = steps
        self.actions = []

    def reset(self):
        self.i = 0
        return np.zeros(33, dtype=np.float32), {"cte": 0.0, "lap_count": 0}

    def step(self, action):
        self.actions.append(float(action[0]))
        info, terminated, truncated = self.steps[self.i]
        self.i += 1
        return np.zeros(33, dtype=np.float32), 0.0, terminated, truncated, info


class CountingHead:
    def __init__(self, steer=0.2):
        self.calls = 0
        self.steer = steer

    def __call__(self, obs):
        self.calls += 1
        return self.steer


def clock():
    state = {"t": 0.0}

    def tick():
        state["t"] += 0.05
        return state["t"]

    return tick


def test_success_episode_metrics():
    env = ScriptedEnv([({"cte": 0.2, "lap_count": 0}, False, False),
                       ({"cte": -0.4, "lap_count": 1, "last_lap_time": 12.5}, False, False)])
    result = run_episode(CountingHead(), env, BY_NAME["nominal"], cte_max=2.0, clock=clock())
    assert result["outcome"] == "success" and result["steps"] == 2
    assert result["avg_abs_cte"] == pytest.approx(0.3)
    assert result["max_abs_cte"] == pytest.approx(0.4)
    assert result["lap_time"] == 12.5 and result["recovery_steps"] is None


def test_crash_and_timeout():
    crash = ScriptedEnv([({"cte": 1.0, "lap_count": 0}, True, False)])
    timeout = ScriptedEnv([({"cte": 0.1, "lap_count": 0}, False, True)])
    assert run_episode(CountingHead(), crash, BY_NAME["nominal"], 2.0, clock=clock())["outcome"] == "crash"
    result = run_episode(CountingHead(), timeout, BY_NAME["nominal"], 2.0, clock=clock())
    assert result["outcome"] == "timeout" and result["lap_time"] is None


def test_offset_start_forces_then_measures_recovery():
    env = ScriptedEnv([
        ({"cte": 0.5, "lap_count": 0}, False, False),
        ({"cte": 1.2, "lap_count": 0}, False, False),  # >= 0.5 * cte_max: hand over
        ({"cte": 0.8, "lap_count": 0}, False, False),
        ({"cte": 0.3, "lap_count": 0}, False, False),  # < 0.2 * cte_max: recovered
        ({"cte": 0.1, "lap_count": 1}, False, False),
    ])
    head = CountingHead()
    result = run_episode(head, env, BY_NAME["offset_start"], cte_max=2.0, direction=-1.0, clock=clock())
    assert env.actions == pytest.approx([-0.6, -0.6, 0.2, 0.2, 0.2])
    assert head.calls == 3
    assert result["recovery_steps"] == 2
    assert result["outcome"] == "success"


def test_plan_runs_interleaves_models():
    conditions = [BY_NAME["nominal"], BY_NAME["low_light"]]
    runs = [(m, c.name, e) for m, c, e in plan_runs(["a", "b"], conditions, 2)]
    assert runs[:4] == [("a", "nominal", 0), ("b", "nominal", 0), ("a", "nominal", 1), ("b", "nominal", 1)]
    assert runs[4] == ("a", "low_light", 0) and len(runs) == 8


class FakeStack:
    """Plays raw env, lag wrapper and latent env at once; every episode is a 1-step success."""

    def __init__(self, fail=False):
        self.throttle = self.delay_steps = self.encoder = self.augmenter = None
        self.fail = fail
        self.closed = False
        self.seen = []

    def reset(self):
        if self.fail:
            raise SimDisconnectedError("boom")
        self.seen.append((self.throttle, self.delay_steps, self.encoder, type(self.augmenter).__name__))
        return np.zeros(33, dtype=np.float32), {"cte": 0.0, "lap_count": 0}

    def step(self, action):
        return np.zeros(33, dtype=np.float32), 0.0, False, False, {"cte": 0.1, "lap_count": 1, "last_lap_time": 3.0}

    def close(self):
        self.closed = True


POLICIES = {
    "m1": SimpleNamespace(encoder="enc1", head=lambda obs: 0.0),
    "m2": SimpleNamespace(encoder="enc2", head=lambda obs: 0.0),
}


def test_run_bench_configures_each_episode_and_resumes(tmp_path):
    stack = FakeStack()
    out = tmp_path / "sim" / "episodes.jsonl"
    conditions = [BY_NAME["low_light"], BY_NAME["low_battery"]]
    kwargs = dict(n_episodes=1, out_path=out, base_throttle=0.25, cte_max=2.0, clock=clock())
    assert run_bench(lambda: (stack, stack, stack), POLICIES, conditions, **kwargs) == 4
    assert stack.seen == [
        (0.25, 0, "enc1", "LowLight"),
        (0.25, 0, "enc2", "LowLight"),
        (pytest.approx(0.175), 3, "enc1", "NoneType"),
        (pytest.approx(0.175), 3, "enc2", "NoneType"),
    ]
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(l["model"], l["condition"], l["episode"], l["outcome"]) for l in lines][0] == ("m1", "low_light", 0, "success")
    assert stack.closed
    assert run_bench(lambda: (stack, stack, stack), POLICIES, conditions, **kwargs) == 0
    assert len(out.read_text().splitlines()) == 4


def test_run_bench_rebuilds_env_after_disconnect(tmp_path):
    stacks = [FakeStack(fail=True), FakeStack()]
    made = []

    def factory():
        made.append(stacks[len(made)])
        return made[-1], made[-1], made[-1]

    out = tmp_path / "episodes.jsonl"
    n = run_bench(factory, POLICIES, [BY_NAME["nominal"]], 1, out, 0.25, 2.0, clock=clock())
    assert n == 2 and len(made) == 2
    assert stacks[0].closed
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sim_bench.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.sim_bench'`

- [x] **Step 3: Write the implementation**

```python
# bench/sim_bench.py
"""Simulator robustness benchmark: every model x condition x episode, one lap each.

One simulator session serves all runs: between episodes the throttle, steering
delay, encoder and image perturbation are swapped in place. Runs are
interleaved (per condition, per episode index, all models) and every model
gets the same random perturbation for a given (condition, episode).
Each finished episode is appended to --out immediately; re-running skips
finished episodes, so a crash or Ctrl+C loses at most one episode.

Model-quality gate first:  python -m bench.sim_bench --conditions nominal
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from bench.conditions import BY_NAME, CONDITIONS, MODELS, Condition
from bench.metrics import oscillation_hz, steering_jerk
from car.policy import Policy
from envs.augment import make_augmenter
from envs.cli import add_env_args, env_kwargs_from_args
from envs.donkey_env import DonkeyLaneEnv
from envs.sim_process import SimDisconnectedError
from envs.wrappers import ActuatorLag, LatentEnv

OFFSET_STEER = 0.6


def run_episode(head, env, condition: Condition, cte_max: float, direction: float = 1.0, laps: int = 1,
                max_offset_steps: int = 30, clock=time.monotonic) -> dict:
    obs, info = env.reset()
    forcing = condition.offset_start
    handover, recovery = None, None
    steers, times, abs_ctes = [], [], []
    step = 0
    while True:
        if forcing and (abs(info.get("cte", 0.0)) >= 0.5 * cte_max or step >= max_offset_steps):
            forcing, handover = False, step
        if forcing:
            steer = OFFSET_STEER * direction
        else:
            steer = head(obs)
            steers.append(steer)
            times.append(clock())
        obs, _, terminated, truncated, info = env.step(np.array([steer], dtype=np.float32))
        step += 1
        cte = abs(info.get("cte", 0.0))
        abs_ctes.append(cte)
        if handover is not None and recovery is None and cte < 0.2 * cte_max:
            recovery = step - handover
        if info.get("lap_count", 0) >= laps:
            outcome = "success"
        elif terminated:
            outcome = "crash"
        elif truncated:
            outcome = "timeout"
        else:
            continue
        return {
            "outcome": outcome,
            "steps": step,
            "avg_abs_cte": float(np.mean(abs_ctes)),
            "max_abs_cte": float(np.max(abs_ctes)),
            "jerk": steering_jerk(steers, times),
            "osc_hz": oscillation_hz(steers, times),
            "lap_time": info.get("last_lap_time") if outcome == "success" else None,
            "recovery_steps": recovery,
        }


def plan_runs(models, conditions, n_episodes: int):
    return [(m, c, e) for c in conditions for e in range(n_episodes) for m in models]


def _key(model: str, condition: str, episode: int) -> tuple:
    return model, condition, int(episode)


def _done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    lines = (json.loads(line) for line in out_path.read_text().splitlines() if line.strip())
    return {_key(r["model"], r["condition"], r["episode"]) for r in lines}


def run_bench(env_factory, policies: dict, conditions, n_episodes: int, out_path, base_throttle: float,
              cte_max: float, laps: int = 1, max_restarts: int = 5, clock=time.monotonic) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out_path)
    todo = [r for r in plan_runs(list(policies), conditions, n_episodes) if _key(r[0], r[1].name, r[2]) not in done]
    stack, restarts, finished = None, 0, 0
    try:
        while finished < len(todo):
            model, condition, episode = todo[finished]
            try:
                if stack is None:
                    stack = env_factory()
                raw, lag, top = stack
                raw.throttle = base_throttle * condition.throttle_scale
                lag.delay_steps = condition.steer_delay
                top.encoder = policies[model].encoder
                rng = np.random.default_rng([zlib.crc32(condition.name.encode()), episode])
                top.augmenter = make_augmenter(condition.augmenter, rng)
                direction = 1.0 if episode % 2 == 0 else -1.0
                result = run_episode(policies[model].head, top, condition, cte_max, direction, laps, clock=clock)
            except SimDisconnectedError as exc:
                if stack is not None:
                    stack[2].close()
                    stack = None
                restarts += 1
                if restarts > max_restarts:
                    raise
                print(f"simulator disconnected ({exc}); restart {restarts}/{max_restarts}")
                continue
            with out_path.open("a") as f:
                f.write(json.dumps({"model": model, "condition": condition.name, "episode": episode, **result}) + "\n")
            finished += 1
            print(f"[{finished}/{len(todo)}] {model:9s} {condition.name:12s} ep {episode}: {result['outcome']}")
    finally:
        if stack is not None:
            stack[2].close()
    return finished


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(parser)
    parser.add_argument("--models", nargs="+", default=[f"{m}=models/{m}/policy.npz" for m in MODELS],
                        help="name=path/to/policy.npz ...")
    parser.add_argument("--conditions", nargs="+", default=[c.name for c in CONDITIONS], choices=list(BY_NAME))
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--out", default="results/sim/episodes.jsonl")
    parser.add_argument("--max-restarts", type=int, default=5)
    args = parser.parse_args()

    env_kwargs = env_kwargs_from_args(args)
    policies = {name: Policy(path) for name, path in (spec.split("=", 1) for spec in args.models)}
    first_encoder = next(iter(policies.values())).encoder

    def factory():
        raw = DonkeyLaneEnv(**env_kwargs)
        lag = ActuatorLag(raw)
        return raw, lag, LatentEnv(lag, first_encoder)

    n = run_bench(factory, policies, [BY_NAME[c] for c in args.conditions], args.n_episodes, args.out,
                  base_throttle=env_kwargs["throttle"], cte_max=env_kwargs["cte_max"], laps=args.laps,
                  max_restarts=args.max_restarts)
    print(f"ran {n} episodes -> {args.out}  (summary: python -m bench.report)")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sim_bench.py -v`
Expected: PASS (6 passed)

- [x] **Step 5: Commit**

```bash
git add bench/sim_bench.py tests/test_sim_bench.py
git commit -m "feat: add resumable simulator robustness benchmark"
```

---

### Task 15: Real-car side: pilot part, trial runner, schedule

**Files:**
- Create: `car/pilot_part.py`, `car/trial.py`, `bench/schedule.py`
- Generate: `car/schedule.csv`
- Modify: `README.md` (append "Raspberry Pi setup")
- Test: `tests/test_pilot_part.py`, `tests/test_trial.py`, `tests/test_schedule.py`

**Interfaces:**
- Consumes: `Policy` (Task 2); `MODELS`, `BY_NAME` (Task 13, schedule and its test only). `car/` files import nothing outside the standard library, numpy, `car.policy` and (lazily) PIL.
- Produces:
  - `RobustPilot(policy_path, throttle: float, log_path, image_dir=None, image_every: int = 5, steer_gain: float = 1.0, policy=None)` — donkeycar part. `run(img, mode, user_steer) -> (steering, throttle)`: `img is None` → `(0.0, 0.0)` and nothing logged. Otherwise acts, logs a row (`time, mode, pilot_steer, user_steer, latency_ms`) to a line-buffered CSV, saves every `image_every`-th frame as `<image_dir>/<frame index 6 digits>.jpg`. Previous steering fed to the policy = its own raw output while mode ≠ `"user"`, else the user's steering. `shutdown()` closes the log. `policy` overrides loading (tests).
  - `car/trial.py`: `INSTRUCTIONS: dict[condition, str]`, `find_trial(schedule_path, trial_id) -> dict`, `ask(prompt, parse, input_fn)`, `parse_yes_no(text) -> bool`, `append_result(path, row)`, `main(argv=None, input_fn=input, popen=subprocess.Popen)`. Results columns: `trial_id, condition, model, round, battery_v, lux, completed, notes, log`. Env vars passed to donkeycar: `BENCH_POLICY`, `BENCH_THROTTLE`, `BENCH_LOG`, `BENCH_IMAGES`, `BENCH_IMAGE_EVERY`. Never prints the model name.
  - `bench/schedule.py`: `BLOCK_ORDER`, `make_schedule(models=MODELS, blocks=BLOCK_ORDER, rounds=5, seed=0) -> list[dict]` (keys `trial_id` 3-digit, `condition`, `model`, `round`), `write_schedule(rows, path)`, CLI `python -m bench.schedule [--out car/schedule.csv]`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_pilot_part.py
import csv

import numpy as np
import pytest

from car.pilot_part import RobustPilot
from car.policy import CAMERA_SHAPE

FRAME = np.zeros(CAMERA_SHAPE, dtype=np.uint8)


class FakePolicy:
    def __init__(self, steer=0.4):
        self.steer = steer
        self.prevs = []

    def act(self, frame, prev_steer):
        self.prevs.append(prev_steer)
        return self.steer


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_run_steers_logs_and_tracks_previous_steering(tmp_path):
    policy = FakePolicy()
    pilot = RobustPilot(None, throttle=0.3, log_path=tmp_path / "log.csv", policy=policy)
    assert pilot.run(FRAME, "local", 0.0) == (pytest.approx(0.4), 0.3)
    pilot.run(FRAME, "user", -0.5)
    pilot.run(FRAME, "local", 0.0)
    pilot.shutdown()
    assert policy.prevs == pytest.approx([0.0, 0.4, -0.5])
    rows = _rows(tmp_path / "log.csv")
    assert [r["mode"] for r in rows] == ["local", "user", "local"]
    assert float(rows[1]["user_steer"]) == pytest.approx(-0.5)
    assert float(rows[0]["pilot_steer"]) == pytest.approx(0.4)
    assert float(rows[0]["latency_ms"]) >= 0.0


def test_missing_frame_is_safe(tmp_path):
    pilot = RobustPilot(None, 0.3, tmp_path / "log.csv", policy=FakePolicy())
    assert pilot.run(None, "local", 0.0) == (0.0, 0.0)
    pilot.shutdown()
    assert _rows(tmp_path / "log.csv") == []


def test_steer_gain_clips_output_but_not_previous_steering(tmp_path):
    policy = FakePolicy(0.4)
    pilot = RobustPilot(None, 0.3, tmp_path / "log.csv", steer_gain=5.0, policy=policy)
    assert pilot.run(FRAME, "local", 0.0)[0] == 1.0
    pilot.run(FRAME, "local", 0.0)
    pilot.shutdown()
    assert policy.prevs[1] == pytest.approx(0.4)


def test_saves_every_nth_frame(tmp_path):
    pilot = RobustPilot(None, 0.3, tmp_path / "log.csv", image_dir=tmp_path / "img", image_every=2, policy=FakePolicy())
    for _ in range(5):
        pilot.run(FRAME, "local", 0.0)
    pilot.shutdown()
    assert sorted(p.name for p in (tmp_path / "img").iterdir()) == ["000000.jpg", "000002.jpg", "000004.jpg"]
```

```python
# tests/test_trial.py
import csv
import json

import pytest

from car.trial import main


def _setup(tmp_path):
    schedule = tmp_path / "schedule.csv"
    with schedule.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "condition", "model", "round"])
        writer.writerow(["007", "shadows", "bc_dr", "2"])
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"throttle": 0.3, "car_dir": str(tmp_path / "mycar"), "image_every": 5}))
    models = tmp_path / "models"
    models.mkdir()
    (models / "bc_dr.npz").touch()
    data = tmp_path / "real"
    argv = ["7", "--schedule", str(schedule), "--settings", str(settings), "--data-dir", str(data), "--models-dir", str(models)]
    return argv, data


class FakePopen:
    instances = []

    def __init__(self, cmd, cwd, env):
        self.cmd, self.cwd, self.env = cmd, cwd, env
        self.waits = 0
        FakePopen.instances.append(self)

    def wait(self):
        self.waits += 1
        if self.waits == 1:
            raise KeyboardInterrupt  # operator's Ctrl+C; donkeycar is still shutting down
        return 0


def test_trial_runs_donkeycar_and_records_result(tmp_path, capsys):
    argv, data = _setup(tmp_path)
    answers = iter(["abc", "8.2", "350", "maybe", "y", "slight wobble"])
    main(argv, input_fn=lambda prompt: next(answers), popen=FakePopen)

    proc = FakePopen.instances[-1]
    assert proc.cmd[-2:] == ["manage.py", "drive"]
    assert proc.cwd == str(tmp_path / "mycar")
    assert proc.env["BENCH_POLICY"].endswith("bc_dr.npz")
    assert proc.env["BENCH_THROTTLE"] == "0.3"
    assert proc.env["BENCH_LOG"].endswith("trial_007.csv")
    assert proc.waits == 2  # kept waiting for a clean donkeycar shutdown after Ctrl+C

    with (data / "trials.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert (row["trial_id"], row["condition"], row["model"], row["round"]) == ("007", "shadows", "bc_dr", "2")
    assert float(row["battery_v"]) == 8.2 and float(row["lux"]) == 350.0
    assert row["completed"] == "1" and row["notes"] == "slight wobble" and row["log"] == "trial_007.csv"
    assert "bc_dr" not in capsys.readouterr().out  # operator stays blind to the model


def test_unknown_trial_exits(tmp_path):
    argv, _ = _setup(tmp_path)
    argv[0] = "99"
    with pytest.raises(SystemExit, match="099"):
        main(argv, input_fn=lambda prompt: "1", popen=FakePopen)


def test_missing_model_file_exits_before_driving(tmp_path):
    argv, _ = _setup(tmp_path)
    (tmp_path / "models" / "bc_dr.npz").unlink()
    count = len(FakePopen.instances)
    with pytest.raises(SystemExit, match="bc_dr.npz"):
        main(argv, input_fn=lambda prompt: "1", popen=FakePopen)
    assert len(FakePopen.instances) == count
```

```python
# tests/test_schedule.py
from collections import Counter

from bench.conditions import BY_NAME, MODELS
from bench.schedule import BLOCK_ORDER, make_schedule
from car.trial import INSTRUCTIONS


def test_schedule_is_balanced_and_blocked():
    rows = make_schedule()
    assert len(rows) == 120
    assert [r["trial_id"] for r in rows] == [f"{i:03d}" for i in range(1, 121)]
    assert [rows[i * 20]["condition"] for i in range(6)] == list(BLOCK_ORDER)
    counts = Counter((r["condition"], r["model"]) for r in rows)
    assert set(counts.values()) == {5} and len(counts) == 24
    for condition in BLOCK_ORDER:
        for rnd in range(1, 6):
            models = [r["model"] for r in rows if r["condition"] == condition and r["round"] == rnd]
            assert sorted(models) == sorted(MODELS)


def test_schedule_is_seeded():
    assert make_schedule(seed=0) == make_schedule(seed=0)
    assert make_schedule(seed=0) != make_schedule(seed=1)


def test_every_condition_has_operator_instructions():
    assert set(INSTRUCTIONS) == set(BY_NAME) == set(BLOCK_ORDER)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pilot_part.py tests/test_trial.py tests/test_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write `car/pilot_part.py`**

```python
# car/pilot_part.py
"""donkeycar part that drives with a benchmark policy and logs every frame.

Added to the car's manage.py (see README, "Raspberry Pi setup"):
  inputs  cam/image_array, user/mode, user/angle
  outputs pilot/angle, pilot/throttle
donkeycar's DriveMode part then uses these outputs only in autopilot
(local) mode; in user mode the operator drives, and the log records it.
"""
import csv
import time
from pathlib import Path

import numpy as np

from car.policy import Policy

LOG_FIELDS = ("time", "mode", "pilot_steer", "user_steer", "latency_ms")


class RobustPilot:
    def __init__(self, policy_path, throttle: float, log_path, image_dir=None, image_every: int = 5,
                 steer_gain: float = 1.0, policy=None):
        self.policy = policy if policy is not None else Policy(policy_path)
        self.throttle = throttle
        self.steer_gain = steer_gain  # ponytail: fixed at 1.0 for every trial; a knob only if steering range differs from sim
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_every = image_every
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", newline="", buffering=1)  # line-buffered: rows survive a hard stop
        self._writer = csv.writer(self._file)
        self._writer.writerow(LOG_FIELDS)
        self._prev = 0.0
        self._frames = 0

    def run(self, img, mode, user_steer):
        if img is None:
            return 0.0, 0.0
        start = time.perf_counter()
        raw = self.policy.act(img, self._prev)
        latency_ms = (time.perf_counter() - start) * 1000.0
        steer = float(np.clip(raw * self.steer_gain, -1.0, 1.0))
        user = float(user_steer or 0.0)
        # In the sim the policy sees its own previous command; when the operator
        # drives, the car's actual previous steering is the user's.
        self._prev = raw if mode != "user" else user
        self._writer.writerow([f"{time.time():.3f}", mode, f"{steer:.4f}", f"{user:.4f}", f"{latency_ms:.2f}"])
        if self.image_dir is not None and self._frames % self.image_every == 0:
            from PIL import Image

            self.image_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(self.image_dir / f"{self._frames:06d}.jpg", quality=90)
        self._frames += 1
        return steer, self.throttle

    def shutdown(self):
        self._file.close()
```

- [x] **Step 4: Write `car/trial.py`**

```python
# car/trial.py
"""Run one real-car benchmark trial on the Pi.

    cd ~/tesla && python -m car.trial 17

Looks up the trial's condition and model in car/schedule.csv, asks for
battery voltage and light level, starts donkeycar with the benchmark pilot,
and after Ctrl+C asks whether the lap was completed. Appends one row to
data/real/trials.csv; the drive log is data/real/trial_<id>.csv.

The model name is never shown, so the operator's takeovers can't be
influenced by knowing which model is driving.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

CAR_DIR = Path(__file__).resolve().parent
ROOT = CAR_DIR.parent
RESULT_FIELDS = ("trial_id", "condition", "model", "round", "battery_v", "lux", "completed", "notes", "log")
INSTRUCTIONS = {
    "nominal": "Room lights on. Full battery.",
    "low_light": "Lights off except the dim lamp on its mark (< 100 lux at the track).",
    "shadows": "Lights on. Low lamp on its mark. Assistant sweeps the cardboard shadow over the marked section as the car passes.",
    "distractors": "Lights on. All 6 distractor objects on their marks.",
    "low_battery": "Lights on. Low battery (2S LiPo 7.3-7.4 V resting; never below 7.0 V).",
    "offset_start": "Lights on. Full battery. Car centre 10 cm toward the outer edge line at the start mark, pointing straight.",
}


def find_trial(schedule_path, trial_id: str) -> dict:
    with open(schedule_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["trial_id"] == trial_id:
                return row
    raise SystemExit(f"trial {trial_id} not found in {schedule_path}")


def ask(prompt: str, parse, input_fn=input):
    while True:
        try:
            return parse(input_fn(prompt).strip())
        except ValueError:
            print("  not understood, try again")


def parse_yes_no(text: str) -> bool:
    if text.lower() in ("y", "yes"):
        return True
    if text.lower() in ("n", "no"):
        return False
    raise ValueError(text)


def append_result(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None, input_fn=input, popen=subprocess.Popen):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_id")
    parser.add_argument("--schedule", default=CAR_DIR / "schedule.csv")
    parser.add_argument("--settings", default=CAR_DIR / "settings.json")
    parser.add_argument("--data-dir", default=ROOT / "data" / "real")
    parser.add_argument("--models-dir", default=ROOT / "models")
    args = parser.parse_args(argv)

    trial_id = f"{int(args.trial_id):03d}"
    trial = find_trial(args.schedule, trial_id)
    settings = json.loads(Path(args.settings).read_text())
    policy_path = Path(args.models_dir) / f"{trial['model']}.npz"
    if not policy_path.exists():
        raise SystemExit(f"missing model file {policy_path}")

    print(f"Trial {trial_id}: condition {trial['condition']}, round {trial['round']}")
    print("  " + INSTRUCTIONS[trial["condition"]])
    battery = ask("battery voltage (V): ", float, input_fn)
    lux = ask("light at the track surface (lux): ", float, input_fn)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"trial_{trial_id}.csv"
    env = {
        **os.environ,
        "BENCH_POLICY": str(policy_path),
        "BENCH_THROTTLE": str(settings["throttle"]),
        "BENCH_LOG": str(data_dir / log_name),
        "BENCH_IMAGES": str(data_dir / f"trial_{trial_id}_img"),
        "BENCH_IMAGE_EVERY": str(settings["image_every"]),
    }
    print("Starting donkeycar. Car on the start mark, then switch to autopilot (local) in the web UI.")
    print("Wheel outside an edge line: switch to user, put the car back at that spot, switch to local.")
    print("After 3 takeovers, or at the finish mark: switch to user, then press Ctrl+C here.")
    proc = popen([sys.executable, "manage.py", "drive"], cwd=os.path.expanduser(settings["car_dir"]), env=env)
    while True:
        try:
            proc.wait()
            break
        except KeyboardInterrupt:
            # donkeycar received the same Ctrl+C; wait for it to stop its parts cleanly.
            continue

    completed = ask("lap completed (y/n): ", parse_yes_no, input_fn)
    notes = input_fn("notes: ").strip()
    append_result(data_dir / "trials.csv", {
        "trial_id": trial_id,
        "condition": trial["condition"],
        "model": trial["model"],
        "round": trial["round"],
        "battery_v": battery,
        "lux": lux,
        "completed": int(completed),
        "notes": notes,
        "log": log_name,
    })
    print(f"recorded trial {trial_id}")


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Write `bench/schedule.py`**

```python
# bench/schedule.py
"""Real-trial schedule: condition blocks in a fixed order (battery drains last),
5 rounds per block, all four models back-to-back in a random order each round.

    python -m bench.schedule            # writes car/schedule.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from bench.conditions import MODELS

BLOCK_ORDER = ("nominal", "distractors", "shadows", "low_light", "offset_start", "low_battery")
FIELDS = ("trial_id", "condition", "model", "round")


def make_schedule(models=MODELS, blocks=BLOCK_ORDER, rounds: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for condition in blocks:
        for rnd in range(1, rounds + 1):
            for model in rng.permutation(list(models)):
                rows.append({"trial_id": f"{len(rows) + 1:03d}", "condition": condition, "model": str(model), "round": rnd})
    return rows


def write_schedule(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="car/schedule.csv")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = make_schedule(rounds=args.rounds, seed=args.seed)
    write_schedule(rows, args.out)
    print(f"wrote {len(rows)} trials to {args.out}")


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_pilot_part.py tests/test_trial.py tests/test_schedule.py -v`
Expected: PASS (10 passed)

- [x] **Step 7: Generate the schedule**

Run: `python -m bench.schedule`
Expected: `wrote 120 trials to car/schedule.csv`

- [x] **Step 8: Append the Pi setup section to `README.md`**

````markdown

## Raspberry Pi setup

The Pi needs only donkeycar (numpy and Pillow come with it); no torch.

1. Copy the car code and the four models:
   ```bash
   ssh pi@<car> 'mkdir -p ~/tesla/models'
   scp -r car pi@<car>:~/tesla/
   for m in sac_clean sac_dr bc_clean bc_dr; do scp models/$m/policy.npz pi@<car>:~/tesla/models/$m.npz; done
   ```
2. In the car app's `manage.py` (e.g. `~/mycar/manage.py`), add this block
   inside `drive()` directly **before** the line `V.add(DriveMode(`:
   ```python
       # --- sim2real robustness benchmark pilot ---
       import os
       if os.environ.get("BENCH_POLICY"):
           import sys
           sys.path.insert(0, os.path.expanduser("~/tesla"))
           from car.pilot_part import RobustPilot
           V.add(RobustPilot(os.environ["BENCH_POLICY"], float(os.environ["BENCH_THROTTLE"]),
                             os.environ["BENCH_LOG"], os.environ.get("BENCH_IMAGES"),
                             int(os.environ.get("BENCH_IMAGE_EVERY", "5"))),
                 inputs=["cam/image_array", "user/mode", "user/angle"],
                 outputs=["pilot/angle", "pilot/throttle"])
   ```
   Without `BENCH_POLICY` set, `manage.py drive` behaves exactly as before.
3. Check numpy is at least 1.20 (for `sliding_window_view`), the policy loads, and it is fast enough (well under 50 ms):
   ```bash
   python -c "import numpy; print(numpy.__version__)"
   cd ~/tesla && python -c "import time, numpy as np; from car.policy import Policy; p = Policy('models/sac_clean.npz'); f = np.zeros((120, 160, 3), np.uint8); p.act(f, 0.0); t = time.perf_counter(); [p.act(f, 0.0) for _ in range(50)]; print((time.perf_counter() - t) / 50 * 1000, 'ms')"
   ```
4. A trial: `cd ~/tesla && python -m car.trial <id>` (ids from `car/schedule.csv`).
5. Copy results back: `scp -r pi@<car>:~/tesla/data/real data/`.
````

- [x] **Step 9: Commit**

```bash
git add car/pilot_part.py car/trial.py car/schedule.csv bench/schedule.py README.md tests/test_pilot_part.py tests/test_trial.py tests/test_schedule.py
git commit -m "feat: add donkeycar pilot part, trial runner and trial schedule"
```

---

### Task 16: Report: tables, tests, figures

**Files:**
- Create: `bench/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `CONDITIONS`, `MODELS`, `NOISE_CONDITIONS` (Task 13); `parse_trial_log`, `trial_outcome`, `wilson_interval` (Task 13); sim JSONL format (Task 14); `trials.csv` format (Task 15).
- Produces:
  - `load_sim(path) -> list[dict]` (each row gains `success: bool`; `[]` if the file is missing)
  - `load_real(path) -> list[dict]` (per trial: `model`, `condition`, `success` (= clean lap), `interventions`, `lap_time`, `early_intervention`, `jerk`, `osc_hz`, `latency_ms_mean`; logs resolved next to `trials.csv`; `[]` if missing)
  - `cell_stats(rows, metrics) -> dict[(model, condition), dict]` — `n`, `successes`, `rate`, `ci`, and `(mean, std)` per metric ignoring `None`/`nan`
  - `compare(rows, model_a, model_b, conditions=NOISE_CONDITIONS) -> dict` — `rate_a`, `rate_b`, `n_a`, `n_b`, `p_success` (Fisher exact), `median_jerk_a`, `median_jerk_b`, `p_jerk` (Mann–Whitney U)
  - `sim_real_spearman(sim_cells, real_cells) -> tuple[float, float, int]` — `(rho, p, n_cells)`; `nan`s below 3 shared cells
  - `write_report(sim_rows, real_rows, out_dir) -> Path` — `report.md`; `real_success.png` when real rows exist; `sim_vs_real.png` when both exist
  - CLI: `python -m bench.report [--sim results/sim/episodes.jsonl] [--real data/real/trials.csv] [--out-dir results]`

- [x] **Step 1: Write the failing test**

```python
# tests/test_report.py
import csv
import json
import math

import pytest

from bench.report import cell_stats, compare, load_real, load_sim, sim_real_spearman, write_report


def _rows(model, condition, successes, n, jerk=1.0):
    return [{"model": model, "condition": condition, "success": i < successes, "jerk": jerk + i, "osc_hz": math.nan}
            for i in range(n)]


def test_cell_stats_counts_rates_and_ignores_nan():
    cells = cell_stats(_rows("sac_clean", "nominal", 3, 4), metrics=("jerk", "osc_hz"))
    cell = cells[("sac_clean", "nominal")]
    assert (cell["n"], cell["successes"], cell["rate"]) == (4, 3, 0.75)
    assert cell["ci"][0] < 0.75 < cell["ci"][1]
    assert cell["jerk"][0] == pytest.approx(2.5)
    assert all(math.isnan(v) for v in cell["osc_hz"])


def test_compare_pools_noise_conditions_only():
    rows = (_rows("sac_dr", "nominal", 0, 5) + _rows("sac_dr", "low_light", 5, 5) + _rows("sac_dr", "shadows", 5, 5)
            + _rows("bc_dr", "nominal", 5, 5) + _rows("bc_dr", "low_light", 0, 5) + _rows("bc_dr", "shadows", 1, 5, jerk=10))
    result = compare(rows, "sac_dr", "bc_dr")
    assert (result["n_a"], result["n_b"]) == (10, 10)
    assert result["rate_a"] == 1.0 and result["rate_b"] == 0.1
    assert 0.0 <= result["p_success"] < 0.01
    assert 0.0 <= result["p_jerk"] <= 1.0


def test_sim_real_spearman():
    sim = {("m", c): {"rate": r} for c, r in [("a", 0.1), ("b", 0.5), ("c", 0.9)]}
    real = {("m", c): {"rate": r} for c, r in [("a", 0.0), ("b", 0.2), ("c", 0.6)]}
    rho, p, n = sim_real_spearman(sim, real)
    assert rho == pytest.approx(1.0) and n == 3
    assert math.isnan(sim_real_spearman({("m", "a"): {"rate": 1}}, {("m", "a"): {"rate": 1}})[0])


def test_load_sim_and_missing_files(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"model": "bc_dr", "condition": "nominal", "episode": 0, "outcome": "crash"}) + "\n")
    assert load_sim(path)[0]["success"] is False
    assert load_sim(tmp_path / "none.jsonl") == []
    assert load_real(tmp_path / "none.csv") == []


def test_load_real_uses_log_and_answer(tmp_path):
    with (tmp_path / "trial_001.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "mode", "pilot_steer", "user_steer", "latency_ms"])
        for i in range(40):
            w.writerow([f"{i * 0.05:.3f}", "user" if i < 5 or i >= 35 else "local", "0.1000", "0.0000", "9.00"])
    with (tmp_path / "trials.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "condition", "model", "round", "battery_v", "lux", "completed", "notes", "log"])
        w.writerow(["001", "nominal", "sac_dr", "1", "8.3", "400", "1", "", "trial_001.csv"])
    [row] = load_real(tmp_path / "trials.csv")
    assert row["success"] and row["interventions"] == 0
    assert row["lap_time"] == pytest.approx(1.5)
    assert row["latency_ms_mean"] == pytest.approx(9.0)


def test_write_report_with_sim_only_and_with_both(tmp_path):
    sim = _rows("sac_clean", "nominal", 8, 10) + _rows("sac_clean", "low_light", 4, 10) + _rows("bc_clean", "low_light", 2, 10)
    for r in sim:
        r.update(avg_abs_cte=0.3, recovery_steps=None)
    out = write_report(sim, [], tmp_path / "sim_only")
    text = out.read_text()
    assert "Simulator" in text and "sac_clean" in text
    real = [dict(r, interventions=0, latency_ms_mean=10.0) for r in sim]
    write_report(sim, real, tmp_path / "both")
    assert (tmp_path / "both" / "real_success.png").exists()
    assert (tmp_path / "both" / "sim_vs_real.png").exists()
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.report'`

- [x] **Step 3: Write the implementation**

```python
# bench/report.py
"""Tables, statistical tests and figures for the paper.

    python -m bench.report [--sim results/sim/episodes.jsonl] [--real data/real/trials.csv] [--out-dir results]

Either input may be missing (e.g. before the real trials). Success means a
completed lap in the simulator, and a clean lap (no interventions) on the car.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from bench.conditions import CONDITIONS, MODELS, NOISE_CONDITIONS  # noqa: E402
from bench.metrics import parse_trial_log, trial_outcome, wilson_interval  # noqa: E402

SIM_METRICS = ("avg_abs_cte", "jerk", "osc_hz", "recovery_steps")
REAL_METRICS = ("interventions", "jerk", "osc_hz", "latency_ms_mean")
COMPARISONS = (
    ("RL vs BC (clean)", "sac_clean", "bc_clean"),
    ("RL vs BC (DR)", "sac_dr", "bc_dr"),
    ("DR vs clean (RL)", "sac_dr", "sac_clean"),
    ("DR vs clean (BC)", "bc_dr", "bc_clean"),
)


def load_sim(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{**r, "success": r["outcome"] == "success"} for r in rows]


def load_real(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        for trial in csv.DictReader(f):
            log = parse_trial_log(path.parent / trial["log"])
            outcome = trial_outcome(log, trial["completed"] == "1")
            rows.append({
                "model": trial["model"],
                "condition": trial["condition"],
                "success": outcome["clean_lap"],
                "interventions": outcome["interventions"],
                "lap_time": outcome["lap_time"],
                "early_intervention": outcome["early_intervention"],
                "jerk": log["jerk"],
                "osc_hz": log["osc_hz"],
                "latency_ms_mean": log["latency_ms_mean"],
            })
    return rows


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def _mean_std(values) -> tuple[float, float]:
    v = _finite(values)
    return (float(np.mean(v)), float(np.std(v))) if v else (math.nan, math.nan)


def cell_stats(rows, metrics) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["condition"]), []).append(r)
    cells = {}
    for key, group in groups.items():
        k, n = sum(bool(r["success"]) for r in group), len(group)
        cells[key] = {"n": n, "successes": k, "rate": k / n, "ci": wilson_interval(k, n),
                      **{m: _mean_std(r.get(m) for r in group) for m in metrics}}
    return cells


def compare(rows, model_a: str, model_b: str, conditions=NOISE_CONDITIONS) -> dict:
    a = [r for r in rows if r["model"] == model_a and r["condition"] in conditions]
    b = [r for r in rows if r["model"] == model_b and r["condition"] in conditions]
    ka, kb = sum(bool(r["success"]) for r in a), sum(bool(r["success"]) for r in b)
    p_success = stats.fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1] if a and b else math.nan
    ja, jb = _finite(r.get("jerk") for r in a), _finite(r.get("jerk") for r in b)
    p_jerk = stats.mannwhitneyu(ja, jb)[1] if ja and jb else math.nan
    return {
        "a": model_a, "b": model_b, "n_a": len(a), "n_b": len(b),
        "rate_a": ka / len(a) if a else math.nan, "rate_b": kb / len(b) if b else math.nan,
        "p_success": float(p_success),
        "median_jerk_a": float(np.median(ja)) if ja else math.nan,
        "median_jerk_b": float(np.median(jb)) if jb else math.nan,
        "p_jerk": float(p_jerk),
    }


def sim_real_spearman(sim_cells: dict, real_cells: dict) -> tuple[float, float, int]:
    keys = sorted(set(sim_cells) & set(real_cells))
    if len(keys) < 3:
        return math.nan, math.nan, len(keys)
    rho, p = stats.spearmanr([sim_cells[k]["rate"] for k in keys], [real_cells[k]["rate"] for k in keys])
    return float(rho), float(p), len(keys)


def _fmt(x, digits: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _ordered(cells: dict) -> list[tuple]:
    return [(m, c.name) for m in MODELS for c in CONDITIONS if (m, c.name) in cells]


def _cell_table(cells: dict, metrics) -> list[str]:
    lines = ["| model | condition | n | success | 95% CI | Δ vs nominal | " + " | ".join(metrics) + " |",
             "|" + "---|" * (6 + len(metrics))]
    for model, condition in _ordered(cells):
        cell = cells[(model, condition)]
        nominal = cells.get((model, "nominal"))
        delta = cell["rate"] - nominal["rate"] if nominal else math.nan
        metric_cols = [f"{_fmt(cell[m][0])} ± {_fmt(cell[m][1])}" for m in metrics]
        lines.append(f"| {model} | {condition} | {cell['n']} | {_fmt(cell['rate'])} | "
                     f"{_fmt(cell['ci'][0])}–{_fmt(cell['ci'][1])} | {_fmt(delta)} | " + " | ".join(metric_cols) + " |")
    return lines


def _comparison_table(rows) -> list[str]:
    lines = ["| comparison (pooled over noise conditions) | success A | success B | Fisher p | median jerk A | median jerk B | Mann–Whitney p |",
             "|---|---|---|---|---|---|---|"]
    for label, a, b in COMPARISONS:
        r = compare(rows, a, b)
        lines.append(f"| {label}: {a} (A) vs {b} (B) | {_fmt(r['rate_a'])} (n={r['n_a']}) | {_fmt(r['rate_b'])} (n={r['n_b']}) | "
                     f"{_fmt(r['p_success'], 4)} | {_fmt(r['median_jerk_a'])} | {_fmt(r['median_jerk_b'])} | {_fmt(r['p_jerk'], 4)} |")
    return lines


def _plot_real_success(cells: dict, path: Path) -> None:
    names = [c.name for c in CONDITIONS]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(MODELS):
        rates = [cells[(model, c)]["rate"] if (model, c) in cells else np.nan for c in names]
        ax.bar(np.arange(len(names)) + (i - 1.5) * width, rates, width, label=model)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("clean-lap rate (real car)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_sim_vs_real(sim_cells: dict, real_cells: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for model in MODELS:
        keys = [k for k in sorted(set(sim_cells) & set(real_cells)) if k[0] == model]
        ax.scatter([sim_cells[k]["rate"] for k in keys], [real_cells[k]["rate"] for k in keys], label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("sim success rate")
    ax.set_ylabel("real clean-lap rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(sim_rows, real_rows, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_cells = cell_stats(sim_rows, SIM_METRICS)
    real_cells = cell_stats(real_rows, REAL_METRICS)
    lines = ["# Sim-to-Real Robustness Benchmark: results", ""]
    for title, rows, cells, metrics in (("Simulator", sim_rows, sim_cells, SIM_METRICS),
                                        ("Real car", real_rows, real_cells, REAL_METRICS)):
        lines += [f"## {title}", ""]
        if not rows:
            lines += ["No data yet.", ""]
            continue
        lines += _cell_table(cells, metrics) + [""] + _comparison_table(rows) + [""]
    if real_rows:
        _plot_real_success(real_cells, out_dir / "real_success.png")
        lines += ["![real clean-lap rate](real_success.png)", ""]
    if sim_rows and real_rows:
        rho, p, n = sim_real_spearman(sim_cells, real_cells)
        _plot_sim_vs_real(sim_cells, real_cells, out_dir / "sim_vs_real.png")
        lines += ["## Sim-to-real agreement", "",
                  f"Spearman ρ = {_fmt(rho)} (p = {_fmt(p, 4)}, {n} model × condition cells)", "",
                  "![sim vs real](sim_vs_real.png)", ""]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", default="results/sim/episodes.jsonl")
    parser.add_argument("--real", default="data/real/trials.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    report = write_report(load_sim(args.sim), load_real(args.real), args.out_dir)
    print(report.read_text())
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS (6 passed)

- [x] **Step 5: Run the whole suite**

Run: `pytest`
Expected: all tests pass (121).

- [x] **Step 6: Commit**

```bash
git add bench/report.py tests/test_report.py
git commit -m "feat: add benchmark report with statistical tests and figures"
```

---

### Task 17: End-to-end smoke run on the real simulator (manual)

**Files:** none (outputs in gitignored `data/smoke/`, `models/smoke_*`, `results/smoke/`).

**Interfaces:**
- Consumes: every CLI from Tasks 6–16 against the real simulator, with the calibrated flags from Task 6 (`<flags>`).
- Produces: confidence the whole pipeline runs before any multi-hour job. Result quality does not matter here.

- [x] **Step 1: Collect a small dataset (2 short scripted runs)**

```bash
python -m collect.collect --mode scripted --out-dir data/smoke --n-frames 400 --cte-sign <sign> --seed 1 <flags>
python -m collect.collect --mode scripted --out-dir data/smoke --n-frames 400 --cte-sign <sign> --seed 2 <flags>
```

Expected: `done: 400 frames written` twice; `wc -l data/smoke/labels.csv` prints 801.

- [x] **Step 2: Train tiny VAEs**

```bash
python -m learn.train_vae --data-dir data/smoke --out-dir models/smoke_vae --epochs 2
python -m learn.train_vae --data-dir data/smoke --out-dir models/smoke_vae_dr --dr --epochs 2
```

Expected: finite losses; `models/smoke_vae/{vae.pth,encoder.npz,recon.png}` exist; the DR `recon.png` top row shows randomized frames, middle row clean.

- [x] **Step 3: Train a tiny BC model**

Run: `python -m learn.train_bc --data-dir data/smoke --vae models/smoke_vae_dr/vae.pth --dr --out models/smoke_bc/policy.npz --epochs 2`
Expected: 2 epoch lines; `models/smoke_bc/policy.npz` exists.

- [x] **Step 4: Train SAC briefly, interrupt once, kill the sim once**

Run: `python -m learn.train_sac --encoder models/smoke_vae/encoder.npz --run-dir models/smoke_sac --total-timesteps 3000 --checkpoint-freq 1000 <flags>`
- When `models/smoke_sac/checkpoints/sac_1000_steps.zip` appears, press Ctrl+C; `pgrep -fa DonkeySim` prints nothing; rerun the same command → it prints `resuming from ...`.
- While it trains again, kill only the simulator: `pkill -9 -f DonkeySim` → within ~10 s the log prints `simulator disconnected (...); restart 1/5` and training continues.
Expected: ends with `saved models/smoke_sac/sac_final.zip and models/smoke_sac/policy.npz`.

- [x] **Step 5: Sim benchmark, interrupted and resumed**

Run: `python -m bench.sim_bench --models sac_clean=models/smoke_sac/policy.npz bc_dr=models/smoke_bc/policy.npz --conditions nominal low_battery offset_start --n-episodes 1 --max-episode-steps 300 --out results/smoke/episodes.jsonl <flags without --max-episode-steps>`
Press Ctrl+C after the first `[1/6]` line, then rerun the same command.
Expected: the rerun starts at `[1/5]`; `results/smoke/episodes.jsonl` ends with 6 lines. Watching the sim window: in `offset_start` the car first swerves to one side before the model steers; in `low_battery` it drives visibly slower.

- [x] **Step 6: Report**

Run: `python -m bench.report --sim results/smoke/episodes.jsonl --real data/none.csv --out-dir results/smoke`
Expected: prints a Simulator table with `sac_clean` and `bc_dr` rows for the three conditions, a comparison table (mostly `—`), and "Real car: No data yet.", without errors.

- [x] **Step 7: Clean up**

```bash
rm -r data/smoke models/smoke_vae models/smoke_vae_dr models/smoke_bc models/smoke_sac results/smoke
```

No commit: this task changes no tracked files. Real data collection, training, the benchmark and the car trials follow `todo.md`.
