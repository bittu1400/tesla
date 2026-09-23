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
python -m bench.sim_bench --conditions nominal --out results/sim/gate.jsonl  # model-quality gate
python -m bench.sim_bench                                     # full sim benchmark
python -m bench.schedule                                      # real-trial schedule -> car/schedule.csv
python -m bench.report                                        # results/report.md
```

On the Pi: `python -m car.trial <id>` (see `todo.md`).
If a crashed run left a simulator running: `pkill -f DonkeySim`.

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
