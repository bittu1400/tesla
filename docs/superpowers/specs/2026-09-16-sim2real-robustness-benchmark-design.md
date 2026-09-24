# Sim-to-Real Robustness Benchmark: RL vs Behavioral Cloning — Design Spec

## Research question

**Main RQ:** How resilient are simulation-trained Deep Reinforcement Learning
(DRL) policies compared to Behavioral Cloning (BC) policies when deployed
zero-shot (no real-world training data) to a physical miniature autonomous
car exposed to real-world visual and mechanical noise?

- **RQ1 (visual noise):** How much do low light, shadows and visual
  distractors reduce lap completion for RL vs BC?
- **RQ2 (hardware noise):** How does low battery (slower motor, weaker and
  slower steering) change steering smoothness and oscillation for RL vs BC?
- **RQ3 (recovery):** Which policy recovers better when the car starts
  off its line?
- **RQ4 (added):** Does domain randomization (DR) in simulation change the
  answer, and does robustness measured in the simulator predict robustness on
  the real car?

Working title: *Robustness Benchmark: Sim-Trained Reinforcement Learning vs.
Behavioral Cloning Under Physical Noise*.

## Constraints

- Results (experiments done) within 7 days, 15 at most.
- Real car: Raspberry Pi running donkeycar (5.x `complete` template), Pi
  camera, PWM steering and ESC, battery-powered. Steering and throttle are
  already calibrated with donkeycar's own tools.
- Laptop: RTX 5070 Laptop GPU (8 GB), Linux (Hyprland), Python 3.12 via `uv`.
- Simulator: Donkey Car Unity sim (`DonkeySimLinux`) through `gym-donkeycar`
  pinned to git commit `a1f4ca6961e17a6929c1f2e883358a1edafb479d` (gymnasium API).
- Everything onboard the Pi runs with numpy only (no torch install on the Pi).

## Experimental design

2 × 2 factorial, one trained model per cell (one seed; time budget):

| | No DR (`clean`) | DR (`dr`) |
|---|---|---|
| **RL** (SAC) | `sac_clean` | `sac_dr` |
| **BC** (supervised) | `bc_clean` | `bc_dr` |

### Parity: the only difference between RL and BC is how the head was trained

All four models share this pipeline:

```
camera 120x160 RGB uint8
  -> preprocess: drop top 40 rows, halve width (average column pairs) -> 3x80x80 float in [0,1]
  -> encoder (frozen, from a VAE): 3 conv layers + linear -> 32-dim latent
  -> observation = [latent (32), previous steering (1)]
  -> head: Linear(33,64)-ReLU-Linear(64,64)-ReLU-Linear(64,1) -> clamp to [-2,2] -> tanh
  -> steering in [-1, 1]
throttle: a fixed constant (not learned)
```

- `sac_clean` and `bc_clean` use the same encoder (`vae_clean`);
  `sac_dr` and `bc_dr` use the same encoder (`vae_dr`).
- The head architecture is exactly SB3 SAC's deterministic actor with
  `net_arch=[64, 64]` and gSDE (its mean is clipped to [-2, 2] before tanh),
  so the SAC actor's weights copy into the head with no change in behaviour.
- **Steering only, fixed throttle.** Throttle-to-speed differs between Unity
  and the real motor, so a learned throttle would add a sim-to-real mismatch
  unrelated to the research question. A fixed throttle also makes low
  battery genuinely slow the car (RQ2), and halves the action space, so SAC
  trains faster.
- Speed is not observed (the real car has no speed sensor).
- **Model-quality gate:** before any robustness test, each model must reach
  ≥ 80% one-lap success in the simulator's nominal condition. If one doesn't,
  fix it (more data, more steps) before testing. Comparing a working model
  with a broken one would say nothing about robustness.

### Domain randomization (DR)

DR is image augmentation in Python (no Unity texture changes: too slow to
build in the time budget).

Training DR (`DomainRandomizer`), per episode (RL) or per sample (VAE/BC):
- brightness × U(0.6, 1.4), contrast × U(0.7, 1.3), saturation × U(0.7, 1.3),
  hue shift U(−8, 8) (OpenCV hue units, 0–180)
- 0–2 static random shadow polygons, darkness × U(0.5, 0.8)
- per frame: Gaussian pixel noise std U(0, 8); 30% chance of 3×3 blur

The DR VAE is a **denoising VAE**: input is the augmented frame, the
reconstruction target is the clean frame. The encoder therefore learns to
ignore lighting and noise.

Evaluation perturbations are **outside** the training ranges (held out), so
a DR model can't pass a test just because it saw that exact condition:
- low light: brightness × 0.35 (training minimum is 0.6) plus noise std 6
- shadows: a moving diagonal dark band (× 0.3), unlike the static, lighter
  training shadows
- distractors: three saturated coloured rectangles; training has none

## Data

One sim dataset, ~12k frames, reused:

- **Scripted driver** (~9k frames): proportional controller on CTE plus
  steering noise and occasional hard swerves. Each frame's **label is the
  clean expert steering** (the controller output without noise or swerve),
  while the *executed* steering includes the noise. BC therefore learns how
  to recover from off-centre positions (noise injection with expert labels,
  as in DART).
- **Human keyboard driving** (~3k frames): label is the executed steering;
  keyboard steering ramps toward its target so labels aren't only −1/0/+1.
- Stored as PNG frames plus `labels.csv` (`frame, steer, prev_steer, episode, source`).
  `prev_steer` is the steering executed on the previous step (0 after reset).

Use of the dataset:
- VAEs: frames only.
- BC: frames + `steer` labels + `prev_steer` inputs. During training, Gaussian
  noise (std 0.1) is added to `prev_steer` to reduce copycat behaviour.
- SAC: no dataset; it learns from reward.

## Training

- **VAE:** conv VAE, latent 32, 30 epochs, β = 1, 90/10 split, best-validation
  weights kept. Outputs `vae.pth` (full model), `encoder.npz` (encoder weights
  for numpy), `recon.png` (preview).
- **BC:** frozen encoder, head trained with MSE for 20 epochs, Adam 1e-3,
  validation split **by episode** (neighbouring frames nearly repeat each
  other). DR models augment every sample.
- **SAC:** SB3 SAC with gSDE, `net_arch=[64, 64]`, gradient updates between
  episodes (the sim runs in real time), crash-resumable from checkpoints.
  ~80k steps per model (≈ 1.5–2 h). DR models get a new randomization each
  episode.
  - Reward per step: `1 − min(|cte|/cte_max, 1)` while moving forward, 0 when
    `forward_vel ≤ 0`; −10 and terminate on a collision or the sim's own
    game-over; truncate (no penalty) at `max_episode_steps`.
- **Export:** every model ends as `models/<name>/policy.npz`, holding encoder
  and head weights in the same format. BC writes it directly; SAC copies
  its actor into the head. A test checks that SB3's deterministic
  `predict` matches the exported numpy head.

## Environments and tracks

- Sim track: chosen on day 1 from the gym-donkeycar tracks for resembling a
  tape-on-floor track (candidates: `donkey-warehouse-v0`,
  `donkey-generated-track-v0`). Default in code: `donkey-generated-track-v0`.
  **As run:** `donkey-warehouse-v0` with `--throttle 0.2 --cte-max 2.5
  --cte-offset -6.9`. Its raw CTE is about -6.9 at the centre of the spawn
  lane, so `--cte-offset` re-centres it.
- Sim camera: 160×120. `--cam-fov` sets the sim camera FOV to match the
  real camera (Unity uses vertical FOV; Pi Camera v2 ≈ 49°, v1 ≈ 41°);
  0 keeps the sim default. The chosen value is used for all models.
  **As run:** the models were trained with the sim default FOV (see "As run").
- Real track: tape on floor, copying the sim track's look (edge line colour,
  centre line colour and dashes) and its lane-width-to-car-width ratio,
  measured from sim screenshots. At least one straight (≥ 1.5 m) with a
  start mark, and left and right corners. Floor area marked for repeatable
  placement of lamps and distractor objects.
- Camera tilt on the real car is set once, so that a real frame, after the
  top-40-row crop, shows about the same road area as a sim frame (checked
  side by side with a frame from a short manual donkeycar recording). Then it is locked (tape or glue).

## Evaluation conditions

Same six condition names in sim and real, so results can be correlated.

| Condition | Simulator | Real car |
|---|---|---|
| `nominal` | no perturbation | room lights on (record lux), full battery |
| `low_light` | brightness × 0.35 + noise | lights off except one dim lamp, < 100 lux at track surface |
| `shadows` | moving dark band | fixed lamp low to the floor at a marked spot, plus an assistant sweeping a cardboard sheet's shadow across a marked section as the car passes |
| `distractors` | 3 coloured rectangles | 6 coloured objects at marked positions 5–10 cm outside the edge lines |
| `low_battery` | throttle × 0.7 and steering delayed 3 steps (≈ 150 ms) | battery at ~20% charge: 2S LiPo resting 7.3–7.4 V (never below 7.0 V); 6-cell NiMH resting ≈ 7.0 V. Measured with a multimeter before each trial |
| `offset_start` | forced steering 0.6 (alternating sides) until \|cte\| ≥ 0.5·cte_max or 30 steps, then policy takes over | car placed with its centre 10 cm toward the outer edge line at the start mark, pointing straight |

Sim noise conditions apply one factor at a time (as do the real ones).

### Sim benchmark
- 4 models × 6 conditions × 10 one-lap episodes = 240 episodes, one sim
  session (models and conditions are swapped in place, no restart).
  **As run:** 5 sims in parallel (`--workers 5`). Each worker is one sim session
  running every 5th planned episode. All workers append to the same file.
- Interleaved order: per condition, per episode index, all four models.
  Every model sees the **same random perturbation** for a given
  (condition, episode index).
- Each episode is appended to `results/sim/episodes.jsonl` as soon as it
  finishes; re-running skips completed episodes (crash-resumable).

### Real benchmark
- 4 models × 6 conditions × 5 trials = 120 one-lap trials.
- Schedule generated with a fixed seed: condition blocks in the order
  nominal, distractors, shadows, low_light, offset_start, low_battery
  (battery drains last). Within a block, 5 rounds, each round running all
  four models in a random order (models run back-to-back under the same
  room conditions).
- Trial procedure (`python -m car.trial <id>` on the Pi):
  1. Shows the condition (never the model: the operator is blind to which
     model drives) and prompts for battery voltage and lux.
  2. Starts donkeycar with the benchmark pilot part. Operator places the car
     at the start mark and switches to autopilot (`local`) in the web UI.
  3. If any wheel crosses outside an edge line, the operator takes over
     (`user` mode), puts the car back at that spot inside the lane and switches
     back to `local`. After 3 interventions the trial ends as not completed.
  4. At the finish mark the operator switches to `user` and presses Ctrl+C.
  5. Prompts: lap completed (y/n) and notes. One row is appended to
     `data/real/trials.csv`.
- Onboard log per trial (`data/real/trial_<id>.csv`): wall time, mode,
  pilot steering, user steering, inference latency (ms). Every 5th camera
  frame is saved as JPEG for figures.

## Metrics

Computed by the same code for sim and real where the signal exists.

| Metric | Sim | Real |
|---|---|---|
| Lap success: sim = lap completed without crash; real = **clean lap** (completed, 0 interventions) | yes | yes |
| Interventions per lap | — | yes |
| Mean \|CTE\| | yes | — |
| Steering jerk: mean \|Δsteer/Δt\| (1/s), pilot-controlled samples only | yes | yes |
| Oscillation frequency (Hz): sign changes of steering minus its 1 s moving average (deadband 0.02) ÷ 2 ÷ duration | yes | yes |
| Lap time (s) | sim clock | first `local` to final takeover, clean laps only |
| Recovery steps (offset_start): steps from handover until \|cte\| < 0.2·cte_max | yes | — |
| Early intervention (offset_start): intervention within 5 s of start | — | yes |
| Inference latency mean / p95 (ms) | — | yes |

### Analysis (`bench/report.py` → `results/report.md` + figures)
- Per model × condition: success rate with Wilson 95% CI, mean ± std of
  jerk, oscillation, CTE, interventions, latency.
- Degradation: success(condition) − success(nominal).
- RL vs BC within each DR level, pooled over the five non-nominal
  conditions: Fisher's exact test on success counts; Mann–Whitney U on jerk.
- DR vs clean within each algorithm: same tests.
- Sim-to-real agreement: Spearman correlation between sim success rate and
  real clean-lap rate across the 24 model × condition cells.
- Figures: grouped bars of real clean-lap rate by condition and model;
  scatter of sim vs real success per cell.
- Stated limitations: 5 real trials per cell (wide intervals; conclusions
  rely on pooled tests), one training seed per model, operator-timed lap
  ends (± 0.5 s), possible BC copycat effect, image-space DR instead of
  simulator-side randomization.

## Code structure

```
Tesla/
  car/                  # runs on the Pi: numpy (+ PIL from donkeycar) only
    policy.py           # preprocess(), numpy Encoder/Head/Policy, conv2d
    pilot_part.py       # RobustPilot donkeycar part: act + log
    trial.py            # one real trial: prompts, launches donkeycar, records outcome
    schedule.csv        # generated by bench.schedule
  envs/
    reward.py           # compute_reward (steering-only lane reward)
    sim_process.py      # launch/kill sim, call timeouts, SimDisconnectedError
    donkey_env.py       # DonkeyLaneEnv: steering-only gym-donkeycar wrapper
    wrappers.py         # ActuatorLag, LatentEnv ([latent, prev_steer] observations)
    augment.py          # DomainRandomizer + held-out eval perturbations
    cli.py              # shared sim flags
  collect/
    drivers.py          # ScriptedDriver (expert labels), keys_to_steer
    dataset.py          # DatasetWriter, read_labels, load_frame
    collect.py          # CLI: scripted / manual driving -> data/drive
  learn/
    nets.py             # torch Encoder, ConvVAE, PolicyHead, npz export
    train_vae.py        # clean or denoising-DR VAE
    train_bc.py         # BC head
    sac_config.py       # SAC hyperparameters
    train_sac.py        # crash-resumable SAC + export to policy.npz
  bench/
    conditions.py       # the six conditions
    metrics.py          # jerk, oscillation, Wilson CI, log parsing
    sim_bench.py        # resumable sim benchmark
    schedule.py         # real-trial schedule -> car/schedule.csv
    report.py           # tables, tests, figures
  scripts/
    check_sim.py        # sim smoke check + calibration
  tests/                # pytest, never starts the simulator
  data/ models/ results/  # gitignored outputs
```

## Error handling

- The env owns the sim process and turns hangs into `SimDisconnectedError`
  (gym-donkeycar otherwise waits forever); SAC training resumes from the
  latest checkpoint; the sim benchmark resumes from its JSONL file; data
  collection appends in batches.
- The car part never crashes the drive loop on a missing frame (returns
  0 steering, 0 throttle).
- Trial rows are written only after the operator answers the prompts, so a
  trial aborted with Ctrl+C at the prompt leaves no half row; its log file
  is overwritten if the trial is re-run.

## Testing

pytest, no simulator and no Pi needed:
- numpy `conv2d`/encoder/head match torch within 1e-4; preprocess shape and crop
- exported SAC head matches `SAC.predict(deterministic=True)`
- reward, sim process management (fake sim script), env wrapper (fake
  gym-donkeycar env), `ActuatorLag`, `LatentEnv`
- augmentation: shapes, determinism for a seed, held-out strength (low light darker than DR minimum)
- drivers (expert label vs executed steering), dataset append/read
- VAE and BC training reduce loss on a tiny dataset
- SAC crash-resume (fake env)
- metrics on synthetic signals (2 Hz sine → ≈ 2 Hz), log parsing
  (interventions, clean lap, lap time), sim-bench episode logic with a
  scripted fake env, resume skipping, schedule balance
- car pilot part with a fake policy file (logging, mode handling)

Manual checks: `scripts/check_sim.py` on the real sim; a smoke run
(tiny dataset → VAE → BC → 2k-step SAC → 1-episode sim bench) before the
long runs; on the Pi, a dry run of `car.trial` with the car on a stand.

## As run (2026-09-23 – 2026-09-24)

Record of what was actually done, where it departs from the design above, and
why. Operational details and commands: `README.md`. Personal runbook with
per-step ticks: `todo.md` (gitignored, on the laptop only).

### Deviations and decisions

| # | Decision | Why | Consequence |
|---|---|---|---|
| 1 | Train on the sim's **default camera FOV** (no `--cam-fov`) | The real camera model (todo 1.4) wasn't checked yet, and the user chose to do all camera-independent work first | If the real camera's FOV differs a lot, re-collect all data and retrain every model, then re-run the gate and benchmark |
| 2 | **No manual (keyboard) data yet**: 9,000 scripted frames (3 × 3,000, seeds 1–3, 13 episodes), not ~12k | Manual driving needs the user at the keyboard | If manual data is added later: retrain only BC, reusing the same VAEs so SAC needn't retrain. Then delete the BC rows from `results/sim/episodes.jsonl` and re-run the benchmark for BC |
| 3 | SAC DR was **not restarted** after the speed fixes were found | It had the same setup as `sac_clean` (Intel rendering, default BLAS threads), which keeps the clean vs DR comparison fair. A restart would save ~10 min at most | Both SAC models trained with a ~30 ms encoder delay per step (BLAS oversubscription). The gate and benchmark ran without it |
| 4 | Gate, benchmark and track tests ran with `OMP_NUM_THREADS=1` and **NVIDIA rendering** (PRIME offload); training ran with Intel rendering | Makes 5 parallel sims possible and removes the encoder delay | NVIDIA vs Intel frames: mean difference 2.1/255; the four models' steering differs by at most 0.017 at the same spot |
| 5 | Benchmark **`--max-episode-steps 2400`** (gate and training: 1200) | SAC laps take 870–1078 steps (chatter, see below); `low_battery` × 0.7 throttle → up to ~1540 steps. A 1200 cap would count them as timeouts, not driving failures | Same cap for all models; crashes still end early |
| 6 | **`--workers N`** added to `bench.sim_bench` (worker ports skip 9092) | The sim runs in real time, so parallel sims are the only big speedup. Every sim also opens a second server on 0.0.0.0:9092 | The gate took 11m40s for 40 episodes, including 3 restarts before the 9092 fix |
| 7 | Sim `step_mode` "synchronous" rejected | The sim sent no frames after a reset | — |
| 8 | Benchmark paused at 194/240 for the track test, then resumed | The user asked to switch tracks | The resume skipped finished episodes; 46 episodes ran in 6m6s |

### Training results

- VAE best val loss: clean 54.00, DR 56.78. Lane edges are visible in `recon.png`, and the DR recon looks clean.
- BC best val MSE: clean 0.0441, DR 0.0464.
- SAC: 80k steps each, ~76 min each (17.4–17.7 steps/s). Final `ep_len_mean`:
  `sac_clean` 737 (at 75k), `sac_dr` 776 (at 78.6k).

### Sim benchmark results (warehouse, 10 episodes per cell)

Gate (nominal, 1200-step cap): all four models 10/10.

Success rate:

| model | nominal | low_light | shadows | distractors | low_battery | offset_start |
|---|---|---|---|---|---|---|
| sac_clean | 1.0 | 0.0 | 0.0 | 0.3 | 1.0 | 0.5 |
| sac_dr | 1.0 | 1.0 | 1.0 | 0.2 | 0.7 | 1.0 |
| bc_clean | 1.0 | 0.0 | 0.0 | 0.7 | 0.7 | 1.0 |
| bc_dr | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 |

Pooled over the five noise conditions (n = 50 each):
- **DR vs clean:** SAC 0.78 vs 0.36 and BC 0.90 vs 0.48, Fisher p < 0.0001 for both.
- **RL vs BC success:** clean 0.36 vs 0.48 (p = 0.31), DR 0.78 vs 0.90 (p = 0.17). Not significant.
- **RL vs BC jerk:** median ~17–19 vs ~1.4–1.6, Mann–Whitney p < 0.0001.

Full tables with CIs, CTE, jerk, oscillation and lap time: `results/report.md`
(gitignored, from `python -m bench.report`).

**SAC steering chatter:** in `nominal`, both SAC models oscillate at ~7.5 Hz
with jerk ~18–20, against BC's ~2 Hz and ~1.5. The steering scrubs speed, so a SAC lap takes ~45–50 s against
~26 s for BC at the same throttle. Nothing in the design penalizes steering
rate, and it was not changed. It matters on the real car (servo wear, RQ2
smoothness) and on hills (see below).

### Extension: unseen tracks and multi-track training (not in the original design)

Added at the user's request (2026-09-24): test the models on sim tracks they
have never seen.

Calibration: `--throttle 0.2`, CTE sign +1. Offset and half-width were read
from frames while steering ±0.25.
- `donkey-mountain-track-v0`: `--cte-offset -3.6 --cte-max 2.2`
- `donkey-generated-roads-v0`: `--cte-offset -4.1 --cte-max 1.4`. This one
  may never register a lap, so it runs with a 1200-step cap, and a timeout
  means "survived 60 s".

Results are steps until leaving the lane, 5 episodes per model:

| Models | Warehouse | Mountain | Generated-roads |
|---|---|---|---|
| bc_clean | laps (benchmark) | 48–49 | 31 |
| bc_dr | laps | 78–79 | 47–53 |
| sac_clean | laps | 325, then 4 × ~4,700–4,900* | 57–62 |
| sac_dr | laps | 66–69 | 58–426 |
| bc_multi_clean (warehouse + mountain data) | 5/5 laps | 417–427 | 30–35 |
| bc_multi_dr (warehouse + mountain data) | 5/5 laps | 460–500 | 32–34 |

\* `sac_clean` stalls within ~200 steps: it flips full-lock left/right almost every step
(39 sign changes in 40 steps). It then sits still until it drifts out of the lane.

Findings:
- **Zero-shot:** no model completes either unseen track. On mountain-track,
  three models leave the lane within 48–79 steps. `bc_clean` does it on the first long
  straight by steering +0.96 across the centre line. Image-space DR does not cover a new road
  texture or line style.
- **Multi-track BC:** the models (`data/drive_multi` = 9k warehouse + 9k scripted
  mountain frames, the mountain frames rendered on NVIDIA) keep warehouse
  performance and get 6–9× further than their warehouse-only versions on mountain-track. They still fail on the
  unseen generated-roads, so two training tracks were not enough.
- **Mountain-track is a poor test for this project:** at throttle 0.2 the car
  slows to ~0.05 at the same bend even with smooth steering. This was seen directly
  for `bc_multi_dr` (steps ~380–500), and the `bc_multi_*` crash steps match it. Most likely
  it's an uphill. At throttle 0.3–0.4, `bc_multi_dr` drove faster than its training data
  and left the lane sooner (by step 100 at 0.4). The project fixes throttle, and the
  real car drives on a flat floor. Also, `info["pos"]` stayed near (7, 1, 0) there even while moving.

### Open decisions for the next session

1. **Flat-track extension** (recommended, **not yet approved**): replace
   mountain-track with flat indoor tracks (`minimonaco`, `circuit-launch`,
   `roboracingleague`, `waveshare`, spawn frames seen). Train on warehouse +
   two of them and hold out the other two. If BC generalizes, decide how to
   train SAC across tracks: the env is one sim and one track, so it needs
   either a vectorized env with one sim per track (SB3 SAC then needs
   `train_freq` in steps instead of per episode) or a scene switch between
   episodes.
2. **SAC chatter:** keep it (it is a result) or add a steering-rate term to the
   reward. Any change means retraining both SAC models, then re-running the gate and benchmark.
3. **Manual data and camera FOV** (deviations 1–2): before or after the real trials?
4. Real-car work (todo 1.4, 1.5, Day 4+) is untouched.

## Out of scope

PPO or other RL algorithms, multiple training seeds, Unity-side domain
randomization, obstacle avoidance, extra sensors, fine-tuning on real data.
