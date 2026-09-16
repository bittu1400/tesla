# Lane-Following RL for Donkey Car — Design Spec

## Goal

Train an RL agent that drives a simulated Donkey Car around a track,
staying in its lane, without crashing. The trained model is intended
to later transfer to a real Donkey Car (Raspberry Pi/Jetson + single
front camera). Obstacle avoidance is a stated future phase — not built
now, but the design must not block it.

## Scope

**In scope (this project, phase 1):**
- Simulated environment via `gym-donkeycar`.
- Camera-only observation (single RGB front camera — the only sensor
  the real car has).
- Lane-following reward using the simulator's cross-track-error (CTE)
  signal.
- VAE-based image compression + SAC policy training.
- Evaluation script reporting lap-completion rate and average CTE.

**Explicitly out of scope (future phase):**
- Obstacle objects / obstacle-avoidance reward.
- Additional sensors (ultrasonic/ToF distance sensor) and sensor fusion.
- Real Donkey Car deployment script and hardware calibration.
- Multi-track generalization / domain randomization tuning.

These are noted as extension points below so phase 1 doesn't need
rework, but no code for them ships in this phase.

## Environment / Tooling

- Python 3.11 virtualenv, separate from system Python 3.14 (torch /
  stable-baselines3 / gym-donkeycar don't support 3.14 yet).
- Key deps: `gym-donkeycar`, `gymnasium`, `stable-baselines3`, `torch`.
- Donkey Car simulator binary (Unity build) downloaded separately per
  gym-donkeycar's install instructions — not a pip package.
- Hardware: local RTX 5070 Laptop GPU (8GB) — sufficient for VAE
  training and SAC on a 32-dim latent observation.

## Project Structure

```
Tesla/
  envs/
    donkey_env.py       # gym-donkeycar wrapper: crop/resize/normalize frame,
                         # reward shaping, episode termination
  vae/
    model.py            # conv autoencoder definition
    collect_frames.py   # drive manually/randomly, dump frames to disk
    train_vae.py        # train autoencoder on collected frames -> vae.pth
  rl/
    train_sac.py        # SAC training loop (stable-baselines3)
    config.py            # hyperparameters
  models/                # saved vae.pth, sac checkpoints (gitignored)
  scripts/
    drive.py            # run a trained model in the sim, watch it drive
    eval.py             # run N episodes headless, report metrics
  requirements.txt
  docs/superpowers/specs/   # this file
```

## Data Flow

1. Raw camera frame (RGB, sim-native resolution) →
2. `donkey_env.py` crop (remove sky/hood if needed) + resize + normalize →
3. VAE encoder → 32-dim latent vector →
4. SAC policy observation = latent vector (+ optionally last action) →
5. SAC outputs continuous action `[steering, throttle]` →
6. Sent to sim, sim returns next frame + `cte` + `hit` + `speed`.

## Reward Function (phase 1)

Per step, while episode active:
```
reward = 1.0 - min(abs(cte) / cte_max, 1.0)   # 1.0 at lane center, 0 at edge
reward += throttle_bonus * speed              # small, discourages standing still
```
Episode terminates when:
- `hit` is true (collision / off-track) → large negative terminal reward.
- `abs(cte) > cte_max` (off lane) → treated as `hit`.
- max episode steps reached → normal (not penalized) termination.

**Future extension point:** obstacle-avoidance reward will add a
penalty term keyed off proximity/hit-with-obstacle, reusing the same
`hit` signal and termination path — no rewrite of this function's
structure, just an added term.

## VAE

- Input: cropped/resized camera frame.
- Small conv encoder/decoder, latent size 32.
- Training data: 5-10k frames from manual or random driving in sim.
- Trained standalone, few epochs, saved as `models/vae.pth`.
- Loaded frozen by `donkey_env.py` during RL training and evaluation.

## RL Training

- Algorithm: SAC (Stable-Baselines3), continuous action space.
- Observation: 32-dim VAE latent (extendable with speed/last-action
  later without changing the VAE).
- Train until reward plateaus or the agent completes a lap without
  `hit` consistently across evaluation episodes; checkpoint
  periodically to `models/`.

## Evaluation ("runs on the lane" success criteria)

`scripts/eval.py` loads a checkpoint, runs N episodes (headless),
reports:
- % episodes completed without `hit`.
- average / max `abs(cte)` per episode.

Success for phase 1 = consistently completes a lap with no collisions
and low average CTE.

## Error Handling / Operational Notes

- gym-donkeycar sim connection can drop; training scripts should catch
  disconnects and retry/reconnect rather than crashing a long run.
- `models/` and any collected frame dumps are gitignored (large
  binary/checkpoint data doesn't belong in git history).

## Testing

- Sanity script: instantiate `donkey_env.py`, take a few random
  actions, assert observation shape (32,) and reward is finite —
  catches wrapper/shape bugs before a long training run.
- VAE: quick reconstruction-loss sanity check after training (loss
  decreased from initial, sample reconstruction visually close).
- No RL correctness test beyond the eval script above (RL agents
  aren't unit-testable in the traditional sense — eval.py is the
  check).

## Future Work (not this phase)

- Obstacle objects in sim + extended reward term.
- Real distance sensor (ultrasonic/ToF) fusion into observation.
- Real Donkey Car deployment: crop/resize pipeline must match sim
  exactly; port `drive.py` to read from Pi camera and publish to the
  Donkey Car's actuator API.
- Domain randomization (lighting/textures) in sim to close the sim-to-real gap.
