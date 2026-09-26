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
  **As run:** the models were trained with the sim default FOV, 90° vertical (see "As run" → "Camera FOV sensitivity").
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

## As run (2026-09-23 – 2026-09-26)

Record of what was actually done, where it departs from the design above, and
why. Operational details and commands: `README.md`. Personal runbook with
per-step ticks: `todo.md` (gitignored, on the laptop only).

### Deviations and decisions

| # | Decision | Why | Consequence |
|---|---|---|---|
| 1 | Train on the sim's **default camera FOV** (no `--cam-fov`; the default is 90° vertical, see "Camera FOV sensitivity") | The real camera model (todo 1.4) wasn't checked yet, and the user chose to do all camera-independent work first | The models fail at 70° and 49° in the sim. Unless the real camera is close to 90° vertical, re-collect all data at its FOV and retrain every model, then re-run the gate and benchmark |
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

### Session 2026-09-26: the user's decisions on the open items

| 2026-09-24 open item | User's decision | What was done |
|---|---|---|
| Flat-track extension | Do everything that can be done in the sim now, so the project is ready when the camera and car are available; all physical work later (paraphrased) | Done, "Extension 2" below |
| SAC steering chatter | Keep it and report it as a result | Nothing retrained. Changing only SAC's reward would also change the RL vs BC comparison |
| Manual data and camera FOV | Physical work later; all sim work now | Manual data still not collected: it needs the user at the keyboard. The FOV question was studied in the sim ("Camera FOV sensitivity" below), and the rebuild steps were prepared |
| Push `master` | Push | `a5ebc1d..c0a6cb1` pushed to `origin/master` |

### Extension 2: flat tracks (2026-09-26)

Goal: train BC on more than one flat track and test it on flat tracks it has
never seen. Mountain-track was a poor test (hill, see above).

**Track screening.** Each candidate got a calibration montage: frames every 5
steps with the raw CTE, at steer 0, +0.25 and -0.25. Then a noise-free PD lap
(`ScriptedDriver` with `noise_std=0`, `swerve_prob=0`) driving on the true CTE checked
whether CTE follows the painted lane all the way round. The lap test found that on
several tracks the sim's CTE path departs from the painted lane in corners: the
car sits on a line while CTE reads well inside the lane. Per-track flags: `README.md`,
"Unseen tracks and multi-track training".

| Track | What the screening showed | Use |
|---|---|---|
| `warren` (taped lines on grass) | CTE matches the lane over the whole lap. PD lap: 557 steps, peak \|CTE\| 1.15 at gain 0.3 and 0.58 at gain 0.6 | training |
| `minimonaco` | In corners CTE is off by about 0.7 (step 160 of a PD run: car on the white edge line, CTE -0.72, half-width ≈ 1.45). With gain 0.6 the PD lap completes within a bound of 1.8 (765 steps, peak 1.65) but not within 1.2 | held out, bound 1.8 |
| `generated-track` | Same kind of drift, seen in frames (car next to the edge line at CTE -0.63). The PD lap completes within 1.8 (446 steps, peak 1.38), not within 1.25 | held out, bound 1.8 |
| `circuit-launch` | Lane taken as the left half (offset -3.0, half-width 3.0). Every PD run ends at steps 525–531 whatever the bound (2.5–4.4): CTE is +0.51 at step 520 (car on the yellow line) and +3.3 to +4.4 at steps 527–531. The car moves at most ~0.9 sideways in 11 steps (forward_vel ≈ 1.6), so the CTE path jumps | not used |
| `roboracingleague` | Raw CTE ≈ +52 at spawn, +steer lowers it (reversed sign). Crossing white lines and boxes on the track (hit `Cube (1)`) | not used |
| `waveshare` | Driving straight, CTE falls from 4.75 to 0 and rises again. At step 30 the three runs (steer 0, +0.25, -0.25) read +0.89, +0.99, -0.99. Tiny mat: at +0.25 the car was off it by step 50 | not used |
| `avc-sparkfun` | Parking lot with red speed bumps and bay markings, no lane | not used |
| `thunderhill` | Scene never loads (`loaded scene` never logged): 2026-09-24 with 5 other sims, 2026-09-26 with 2 | not used |

**Decisions** (numbered on from the table above):

| # | Decision | Why | Consequence |
|---|---|---|---|
| 9 | Screen tracks with a calibration montage and a noise-free PD lap on the true CTE | Frames give offset and width only near the spawn point. The lap shows whether CTE follows the painted lane all the way round | Found the corner drift on minimonaco and generated-track, and the CTE jump on circuit-launch |
| 10 | Keep the warehouse lane on every track: the lane with the white edge line on the left and the yellow dashed centre line on the right | Warehouse, warren, minimonaco and generated-track all spawn the car in such a lane. Driving another way on one track (e.g. on circuit-launch's yellow line) would give BC contradictory rules | `--cte-offset` is the raw CTE at the centre of that lane, not the sim path |
| 11 | Train only on tracks whose CTE matches the painted lane (warren); use drifting tracks only as held-out tests | BC labels come from CTE, so drift would teach crossing the lines in corners | One new training track, not the two proposed on 2026-09-24 |
| 12 | Held-out bound 1.8 on minimonaco and generated-track: the smallest value tried at which the PD driver on the true CTE completes a lap (1.2 and 1.25 failed) | A bound the expert can't meet would count every model as crashing. The painted half-width (~1.5) is too tight because of the drift | A held-out crash means leaving the road area, not just the lane |
| 13 | Warren PD gains doubled: `--gain 0.6 --d-gain 8` (new `collect.collect` flags; defaults 0.3 and 4.0 = the warehouse values) | Warren's half-width (~1.4) is about half the warehouse's (3). With gain 0.3 the noise-free lap peaked at \|CTE\| 1.15, just inside the 1.2 bound; with 0.6 it peaked at 0.58 | Warren labels steer harder per unit of CTE. At the lane edge they give about the same steering as warehouse (0.6 × 1.4 ≈ 0.84 vs 0.3 × 3 = 0.9) |
| 14 | Warren collection with `--cte-max 1.4` (evaluation: 1.2) | At 1.2, a 600-frame trial ended its episodes at 213 and 337 steps (lap ≈ 557), so the start of the track would dominate the data. At 1.4 (about the painted edge) a 1,500-frame trial ran 490 and 943 | 20 episodes, 25–1,200 steps, mean 450 (warehouse data: 13 episodes, 51–2,000, mean 692). Evaluation keeps the warehouse ratio (0.83 × half-width) |
| 15 | Warren collection with `--max-episode-steps 1200` | Same cap as training and the gate | 1 of 20 episodes reached it. The warehouse data used the default, 2000 |
| 16 | Two new model sets, BC only, default hyperparameters: `flat` = warehouse + warren and `tri` = warehouse + mountain + warren, each clean and DR | `flat` tests flat tracks without the hilly mountain data; `tri` tests whether a third track helps. BC trains in minutes | `models/{vae,bc}_{flat,tri}_{clean,dr}` |
| 17 | `train_vae` and `train_bc` take several `--data-dir` values (concatenated), instead of a merged copy | No copying or renumbering. The 2026-09-24 merge script lived in a session scratchpad that is gone | `data/drive_multi` stays a merged copy (warehouse + mountain). A single dir gives the same frame order as before |
| 18 | No SAC across tracks | BC on 2–3 tracks did not transfer to unseen ones, so a multi-sim SAC setup (a code change: one sim per track, or a scene switch) is unlikely to pay off | 2026-09-24 open decision 1 closed |
| 19 | Evaluation: `nominal`, 5 episodes per model and track, throttle 0.2, `<speed env>`, 4–5 workers. Caps: warehouse 2400, warren, minimonaco and generated-track 3000, mountain 6000, roads 1200 | Same protocol as the first extension. The caps sit well above the lap lengths | The older 6 models ran on the 3 new tracks (`results/sim/track_*.jsonl`); the 4 new models ran on all 6 tracks (`results/sim/flat_*.jsonl`) |
| 20 | The DR negative transfer on warehouse (see findings) was traced once and not investigated further | One training seed per model. A fix (more data, reweighting) would be a new experiment | Recorded as a finding, with that caveat |

**Data and models.**
- `data/drive_warren`: 9,000 scripted frames (3 × 3,000, seeds 1–3, 20 episodes, ~9 min). Rendered on
  NVIDIA, like the mountain frames; the warehouse frames were rendered on Intel.
- `flat` (`--data-dir data/drive data/drive_warren`, 18k frames): VAE val loss 80.62 / 83.01 (clean / DR),
  BC val MSE 0.0393 / 0.0448.
- `tri` (`--data-dir data/drive_multi data/drive_warren`, 27k frames): VAE 75.01 / 77.56, BC 0.0416 / 0.0315.
- The VAE losses are higher than warehouse-only (54.00 / 56.78) because grass texture is hard to
  reconstruct. Both lane lines are sharp in `recon.png`.
- VAEs 30 epochs and BC 20 epochs, as before. The 4 VAEs took ~9 min and the 4 BC heads ~6 min, two at a time.

**Results.** Laps out of 5, then steps per episode as median (min–max); a crash means leaving the lane (road, on the held-out tracks):

| model | warehouse | warren | minimonaco | generated-track | mountain | roads |
|---|---|---|---|---|---|---|
| bc_clean | laps (benchmark) | 0/5, 55 (53–61) | 0/5, 99 (98–142) | 0/5, 101 (45–112) | see the first extension | see the first extension |
| bc_dr | laps | 0/5, 97 (89–113) | 0/5, 157 (156–162) | 0/5, 42 (41–108) | | |
| sac_clean | laps | 0/5, 381 (253–394) | 0/5, 72 (48–77) | 0/5, 32 (31–150) | | |
| sac_dr | laps | 0/5, 66 (60–68) | 0/5, 283 (261–335) | 0/5, 83 (76–103) | | |
| bc_multi_clean | 5/5 | 0/5, 51 (47–52) | 0/5, 76 (64–78) | 0/5, 94 (65–95) | | |
| bc_multi_dr | 5/5 | 0/5, 87 (78–113) | 0/5, 149 (98–150) | 0/5, 69 (68–132) | | |
| bc_flat_clean | 5/5 | 5/5 | 0/5, 52 (44–71) | 0/5, 155 (55–197) | 0/5, 99 (99–100) | 0/5, 63 (60–180) |
| bc_flat_dr | **1/5** (4 crashes at 384–385) | 5/5 | 0/5, 85 (67–147) | 0/5, 65 (41–83) | 0/5, 65 (64–67) | 0/5, 2 of 5 reached the 1200-step cap (survived 60 s); 745 (250–1200) |
| bc_tri_clean | 5/5 | 5/5 | 0/5, 129 (123–142) | 0/5, 82 (40–205) | 0/5, 460 (415–466) | 0/5, 316 (143–711) |
| bc_tri_dr | **0/5** (383–384) | 5/5 | 0/5, 153 (148–249) | 0/5, 86 (54–166) | 0/5, 379 (376–412) | 0/5, 300 (163–705) |

PD reference laps: warren 557, minimonaco 765, generated-track 446 steps. Each evaluation set took 8–10 min (older models: 90 episodes, 4 workers; new models: 120 episodes, 5 workers).

Findings:
- **Trained tracks vs unseen tracks.** All four new models lap warren 5/5, and the clean ones also lap warehouse
  (the DR ones don't; see negative transfer below). No model laps a track it wasn't trained on: the older models
  leave warren's lane within 47–394 steps.
- **More training tracks barely help on unseen tracks.** On minimonaco the new models
  last 44–249 steps, against 48–335 for the older ones. On generated-track they last 40–205, against 31–150.
  Generated-roads is the exception: `tri` lasts 143–711 steps against 30–35 for `multi`, and `bc_flat_dr` survived
  the full 60 s twice, a first on that track.
  On mountain, `tri` (376–466) equals `multi` (417–500), and `flat`, without mountain data, fails early (64–100).
  With 2–3 training tracks, BC plus image-space DR does not transfer zero-shot to a new track.
- **Negative transfer under DR:** `bc_flat_dr` and `bc_tri_dr` lose the warehouse lap
  (1/5 and 0/5) that `bc_dr` and `bc_multi_dr` completed. Both crash at step ~384, in a
  right-hand corner. Traced for `bc_tri_dr`: from step 372 it steers only +0.26 to +0.46, under-steers and
  drifts out to the left (CTE -0.71 at step 364, -2.56 at 383). In the same sim session `bc_multi_dr` and
  `bc_tri_clean` lapped. The clean versions make that corner. One seed per model (decision 20).

### Camera FOV sensitivity (2026-09-26)

Why: the real camera's FOV is the open item that decides whether the models can go to
the car at all, and it can be studied in the sim.

- **The sim's default camera is 90° vertical FOV.** A frame at spawn with `--cam-fov 90` matches the
  default frame: mean difference 3.5/255, against 33.6–48.6 for 41, 49, 60, 70, 80, 100 and 110.
  Every model so far was trained at 90°.
- **The models don't tolerate a narrower FOV.** The four main models on warehouse, `nominal`, 5 episodes each
  (`results/sim/fov_{70,49}.jsonl`):
  - `--cam-fov 70`: 0/20 laps, crashes at 51–213 steps.
  - `--cam-fov 49`: 0/20 laps, crashes at 49–306 steps.
  49° is the Pi Camera v2's vertical FOV; 70° is a midpoint.
- **What this means for the real car:**
  - Pi Camera v2 (62.2° × 48.8°, H × V) or v1 (53.5° × 41.4°): re-collect every
    dataset at that vertical FOV and retrain every model.
  - A wide-angle camera near 90° vertical may be close enough, but check a real frame next to a sim
    frame first. A fisheye lens also distorts the image, and the pipeline doesn't model that: `DonkeyLaneEnv`
    sends only `fov`, and gym-donkeycar's `fish_eye_x`/`fish_eye_y` camera settings are unused.
  - Camera specs often quote horizontal or diagonal FOV. The sim uses vertical.
- **The rebuild is mechanical.** The calibration (offsets, bounds, throttle, PD gains) doesn't depend on
  FOV. Steps: `todo.md`, "Camera FOV rebuild". Everything goes into `*_fov<F>` dirs, so the 90° models
  and results stay the record. Estimate: ~2.5 h if both SAC runs train at once (not tried before),
  ~4 h one after the other.

### Open decisions (after 2026-09-26)

1. **Real camera FOV** (todo 1.4): the one thing on the sim side that blocks the real trials.
   Near 90° vertical: keep the models. Otherwise: rebuild at that FOV.
2. **Manual data** (deviation 2): needs the user at the keyboard. Best done in the same pass as a FOV
   rebuild, because a rebuild re-collects all data anyway.
3. Real-car work (todo 1.4, 1.5, Day 4+) has not started.

## Out of scope

PPO or other RL algorithms, multiple training seeds, Unity-side domain
randomization, obstacle avoidance, extra sensors, fine-tuning on real data.
