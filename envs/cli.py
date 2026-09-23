"""Command-line flags shared by every script that builds a DonkeyLaneEnv.
Use identical values for collection, training and the benchmark."""
import argparse
import os


def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exe-path",
        default=os.environ.get("DONKEY_SIM_PATH"),
        help="simulator binary, or 'remote' for a sim you started yourself (default: $DONKEY_SIM_PATH)",
    )
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--env-name", default="donkey-generated-track-v0")
    parser.add_argument("--cte-max", type=float, default=2.0, help="lane half-width; beyond it the episode ends")
    parser.add_argument("--cte-offset", type=float, default=0.0, help="raw sim CTE at the lane centre (see check_sim)")
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--throttle", type=float, default=0.25, help="fixed sim throttle")
    parser.add_argument("--cam-fov", type=int, default=0, help="sim camera FOV in degrees; 0 keeps the sim default")


def env_kwargs_from_args(args: argparse.Namespace) -> dict:
    if not args.exe_path:
        raise SystemExit("error: pass --exe-path or set DONKEY_SIM_PATH")
    return {
        "exe_path": args.exe_path,
        "port": args.port,
        "env_name": args.env_name,
        "cte_max": args.cte_max,
        "cte_offset": args.cte_offset,
        "max_episode_steps": args.max_episode_steps,
        "throttle": args.throttle,
        "cam_fov": args.cam_fov,
    }
