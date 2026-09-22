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
