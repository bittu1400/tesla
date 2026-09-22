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
