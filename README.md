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
