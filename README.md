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
`--cte-offset`, `--cam-fov`, `--max-episode-steps`: use the same values
everywhere. Calibrated for the warehouse track: `--env-name donkey-warehouse-v0
--throttle 0.2 --cte-max 2.5 --cte-offset -6.9` (raw sim CTE is about -6.9 at
the centre of the lane the car spawns in, so CTE is re-centred on that lane).
The trained models use the sim's default camera FOV (no `--cam-fov`; that is 90° vertical),
because the real camera had not been checked yet. In the sim they fail at 70° and at 49°
(0/20 laps each), so a real camera far from 90° vertical means rebuilding everything at its
FOV: see "Rebuilding at another camera FOV" below.

```bash
python -m scripts.check_sim                                   # sim check + calibration
python -m collect.collect --mode scripted --n-frames 3000     # driving data (repeat, then --mode manual)
python -m learn.train_vae --out-dir models/vae_clean
python -m learn.train_vae --out-dir models/vae_dr --dr
python -m learn.train_bc --vae models/vae_clean/vae.pth --out models/bc_clean/policy.npz
python -m learn.train_bc --vae models/vae_dr/vae.pth --dr --out models/bc_dr/policy.npz
python -m learn.train_sac --encoder models/vae_clean/encoder.npz --run-dir models/sac_clean --max-episode-steps 1200
python -m learn.train_sac --encoder models/vae_dr/encoder.npz --dr --run-dir models/sac_dr --max-episode-steps 1200
python -m bench.sim_bench --conditions nominal --out results/sim/gate.jsonl --workers 5 --max-episode-steps 1200  # model-quality gate
python -m bench.sim_bench --workers 5 --max-episode-steps 2400  # full sim benchmark
python -m bench.schedule                                      # real-trial schedule -> car/schedule.csv
python -m bench.report                                        # results/report.md
```

The benchmark uses `--max-episode-steps 2400`, not 1200. A SAC lap takes
870–1078 steps: its steering chatters, which slows the car. `low_battery`
multiplies throttle by 0.7, so a SAC lap there needs up to ~1540 steps, and a
1200 cap would count those episodes as timeouts however well it drove. The
same cap applies to every model, and crashes still end episodes early.

On the Pi: `python -m car.trial <id>` (see `todo.md`).
If a crashed run left a simulator running: `pkill -f DonkeySim`.

### Speed

The sim runs in real time at 20 steps/s, so sim time sets the pace, not the GPU.
SAC training reached 17.4–17.7 steps/s, the gap being mostly resets (~1.2 s of fixed
sleeps in gym-donkeycar per episode) and gradient steps (~5.4 ms each on the GPU).

- `--workers N` (sim benchmark only) runs N sims at once on ports `--port`, `--port+1`, ...,
  skipping 9092. Every sim also opens a second server on 0.0.0.0:9092, so a sim launched on
  9092 fails to connect. Five sims on the RTX 5070 each keep 20 steps/s. Use a worker count
  that doesn't divide 4 (e.g. 5), so each worker gets a mix of models: worker i runs
  every N-th planned episode.
- `export OMP_NUM_THREADS=1`: with default BLAS threading the numpy encoder
  took ~31 ms per frame (measured while SAC training ran) instead of ~0.9 ms,
  which delays every steering command and keeps ~22 cores busy.
- `export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`: without it
  the sim renders on the Intel iGPU (see `unitylog_<port>.txt`, line `Renderer:`),
  where four sims held ~20 steps/s but six dropped to 16–18 (skipped frames change the
  driving dynamics). NVIDIA frames differ from Intel ones by
  2.1/255 on average (mostly background), and the four models' steering by at most 0.017.
- The sim's `step_mode` "synchronous" message was tried: after a reset the sim sent
  no frames, so it is not used.

The four models were trained without these two exports (Intel rendering, default
threads). The gate, the benchmark, the mountain and warren data collection and every track
test below used both. Use both for every new run, training included.

### Unseen tracks and multi-track training (extension)

The four models have only seen the warehouse. Calibrations for other tracks
(`--throttle 0.2`, CTE sign +1 on all). Offset and half-width were read from
frames while steering ±0.25. The tracks added on 2026-09-26 (warren, minimonaco, generated-track)
were also checked with a noise-free PD lap (`ScriptedDriver` with `noise_std=0`); mountain and
roads (2026-09-24) were not. Every lane used has the warehouse layout:
a white edge line on the left and a yellow dashed centre line on the right.

| Track | `--cte-offset` | `--cte-max` | Use | Notes |
|---|---|---|---|---|
| `donkey-warren-track-v0` | -4.2 | 1.2 (collect: 1.4) | training | Taped lines on grass. Half-width ≈ 1.4. PD lap ≈ 557 steps. Collect with `--gain 0.6 --d-gain 8` |
| `donkey-mountain-track-v0` | -3.6 | 2.2 | training (multi, tri) | Hilly. At throttle 0.2 the car slows to ~0.05 at one bend even with smooth steering (most likely an uphill) |
| `donkey-minimonaco-track-v0` | -2.15 | 1.8 | held out | Half-width ≈ 1.45, but in corners CTE is off the painted lane by ~0.7. 1.8 is the smallest bound tried at which a PD lap (gain 0.6) completes (1.2 failed): lap ≈ 765 steps, peak \|CTE\| 1.65 |
| `donkey-generated-track-v0` | -4.1 | 1.8 | held out | Half-width ≈ 1.5, same kind of drift. A PD lap (gain 0.6) completes within 1.8 (1.25 failed): lap ≈ 446 steps, peak \|CTE\| 1.38 |
| `donkey-generated-roads-v0` | -4.1 | 1.4 | held out | Narrow lane (half-width ≈ 1.7). May never register a lap, so run it with `--max-episode-steps 1200` and read a timeout as "survived 60 s" |

Not usable:
- `circuit-launch`: every PD run ends at steps 525–531, whatever the bound. CTE rises from +0.51 to +3.3–4.4 in about 11 steps, far faster than the car moves sideways.
- `roboracingleague`: raw CTE ≈ +52 with the sign reversed, crossing lines and boxes on the track.
- `waveshare`: CTE reaches 0 while driving straight across a curve, and its sign flips with small steering changes.
- `avc-sparkfun`: parking lot with speed bumps, no lane.
- `thunderhill`: the scene never loads (2 attempts).

Screening method for a new track:
1. Drive it with `--cte-max 1000` at steer 0 and ±0.25, and note the raw CTE where the camera centre crosses the yellow centre line and the white edge line: offset = midpoint, half-width = half the distance.
2. Run a noise-free PD lap on the true CTE (`ScriptedDriver(rng, gain=0.6, d_gain=8, noise_std=0, swerve_prob=0)`) and check in its frames that the car stays in the painted lane all the way round. Only such tracks give correct BC labels.

The narrow lanes need about twice the warehouse PD gains: with `--gain 0.3`, a
corner that needs steer 0.3 settles at |CTE| ≈ 1, most of a 1.4 half-width.
`collect.collect` takes `--gain` and `--d-gain` (defaults 0.3 and 4.0, the warehouse values).
The warren data (`data/drive_warren`, 9k frames, ~9 min) came from:

```bash
for s in 1 2 3; do python -m collect.collect --mode scripted --n-frames 3000 --seed $s --out-dir data/drive_warren --env-name donkey-warren-track-v0 --throttle 0.2 --cte-max 1.4 --cte-offset -4.2 --gain 0.6 --d-gain 8 --max-episode-steps 1200; done
```

`train_vae` and `train_bc` take several `--data-dir` values and concatenate
them, so a multi-track dataset needs no copying or renumbering. Collect each
track into its own dir, one collector per dir (parallel collectors in one dir
race on frame numbers). Never pass overlapping dirs: `data/drive_multi` already
contains every `data/drive` frame, so `--data-dir data/drive data/drive_multi` would train on them twice.

| Models | `--data-dir` |
|---|---|
| `{vae,bc}_multi_{clean,dr}` | `data/drive_multi` (warehouse + mountain, 18k frames; a merged copy) |
| `{vae,bc}_flat_{clean,dr}` | `data/drive data/drive_warren` (warehouse + warren, 18k) |
| `{vae,bc}_tri_{clean,dr}` | `data/drive_multi data/drive_warren` (all three, 27k) |

Evaluation (`nominal`, 5 episodes per model): `bench.sim_bench --conditions nominal --n-episodes 5
--workers 5 --models name=path ... --throttle 0.2` plus the track's flags, with `--max-episode-steps`
2400 (warehouse), 3000 (warren, minimonaco, generated-track), 6000 (mountain) or 1200 (roads).

Results and decisions: the design spec, section "As run". Raw episodes:
`results/sim/{mountain,roads,multi_*,track_*,flat_*}.jsonl`.

### Rebuilding at another camera FOV

Only needed if the real camera's vertical FOV is far from 90° (Pi Camera v2 ≈ 49°, v1 ≈ 41°;
camera specs often give horizontal or diagonal FOV). Everything goes into `*_fov$F`
dirs, so the 90° models and results stay intact. Run it in one terminal from the repo root, with
the venv active and `DONKEY_SIM_PATH` set:

```bash
F=49   # the real camera's vertical FOV
export OMP_NUM_THREADS=1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
FLAGS="--env-name donkey-warehouse-v0 --throttle 0.2 --cte-max 2.5 --cte-offset -6.9 --cam-fov $F"
python -m scripts.check_sim $FLAGS --steps 60 --save-frame data/fov_check.png    # compare with a real frame
for s in 1 2 3; do python -m collect.collect --mode scripted --n-frames 3000 --cte-sign 1 --seed $s --out-dir data/drive_fov$F $FLAGS; done
python -m collect.collect --mode manual --n-frames 3000 --out-dir data/drive_fov$F $FLAGS   # optional, you at the keyboard
python -m learn.train_vae --data-dir data/drive_fov$F --out-dir models/fov$F/vae_clean
python -m learn.train_vae --data-dir data/drive_fov$F --out-dir models/fov$F/vae_dr --dr
python -m learn.train_bc --data-dir data/drive_fov$F --vae models/fov$F/vae_clean/vae.pth --out models/fov$F/bc_clean/policy.npz
python -m learn.train_bc --data-dir data/drive_fov$F --vae models/fov$F/vae_dr/vae.pth --dr --out models/fov$F/bc_dr/policy.npz
python -m learn.train_sac --encoder models/fov$F/vae_clean/encoder.npz --run-dir models/fov$F/sac_clean $FLAGS --max-episode-steps 1200
python -m learn.train_sac --encoder models/fov$F/vae_dr/encoder.npz --dr --run-dir models/fov$F/sac_dr $FLAGS --max-episode-steps 1200 --port 9093
M="sac_clean=models/fov$F/sac_clean/policy.npz sac_dr=models/fov$F/sac_dr/policy.npz bc_clean=models/fov$F/bc_clean/policy.npz bc_dr=models/fov$F/bc_dr/policy.npz"
python -m bench.sim_bench --models $M --conditions nominal --out results/sim_fov$F/gate.jsonl --workers 5 $FLAGS --max-episode-steps 1200
python -m bench.sim_bench --models $M --out results/sim_fov$F/episodes.jsonl --workers 5 $FLAGS --max-episode-steps 2400
python -m bench.report --sim results/sim_fov$F/episodes.jsonl --out-dir results/fov$F
```

Put `systemd-inhibit --what=idle:sleep` in front of the SAC and benchmark lines so the laptop
doesn't sleep; SAC resumes from its last checkpoint if re-run with the same command. The two SAC
runs can train at the same time (different ports, two terminals with the same exports), but that
has not been tried. Estimate: ~2.5 h with them in parallel, ~4 h one after the other. Then copy the `models/fov$F`
policies to the Pi (below).

## Raspberry Pi setup

The Pi needs only donkeycar (numpy and Pillow come with it); no torch.

1. Copy the car code and the four models (after a FOV rebuild, take them from `models/fov<F>/$m/policy.npz` instead):
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
4. Save one frame from the car's camera and compare it to a sim frame to confirm the car isn't feeding BGR where the policy expects RGB (the track's yellow centre line will look blue if swapped):
   ```bash
   cd ~/tesla && python -c "from picamera2 import Picamera2; from PIL import Image; cam = Picamera2(); cam.configure(cam.create_still_configuration(main={'size': (160, 120), 'format': 'RGB888'})); cam.start(); Image.fromarray(cam.capture_array('main')).save('cam_frame.jpg')"
   scp pi@<car>:~/tesla/cam_frame.jpg .
   ```
   Open `cam_frame.jpg` next to a sim frame (e.g. from `collect.collect`) and check the centre line is yellow in both.
   The same pair also shows whether the camera's field of view and tilt match the sim's (see "Rebuilding at another camera FOV").
5. A trial: `cd ~/tesla && python -m car.trial <id>` (ids from `car/schedule.csv`).
6. Copy results back: `scp -r pi@<car>:~/tesla/data/real data/`.
