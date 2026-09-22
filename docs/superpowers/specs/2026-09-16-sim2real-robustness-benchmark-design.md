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
- Sim camera: 160×120. `--cam-fov` sets the sim camera FOV to match the
  real camera (Unity uses vertical FOV; Pi Camera v2 ≈ 49°, v1 ≈ 41°);
  0 keeps the sim default. The chosen value is used for all models.
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

## Out of scope

PPO or other RL algorithms, multiple training seeds, Unity-side domain
randomization, obstacle avoidance, extra sensors, fine-tuning on real data.
