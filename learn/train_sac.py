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
