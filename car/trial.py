"""Run one real-car benchmark trial on the Pi.

    cd ~/tesla && python -m car.trial 17

Looks up the trial's condition and model in car/schedule.csv, asks for
battery voltage and light level, starts donkeycar with the benchmark pilot,
and after Ctrl+C asks whether the lap was completed. Appends one row to
data/real/trials.csv; the drive log is data/real/trial_<id>.csv.

The model name is never shown, so the operator's takeovers can't be
influenced by knowing which model is driving.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

CAR_DIR = Path(__file__).resolve().parent
ROOT = CAR_DIR.parent
RESULT_FIELDS = ("trial_id", "condition", "model", "round", "battery_v", "lux", "completed", "notes", "log")
INSTRUCTIONS = {
    "nominal": "Room lights on. Full battery.",
    "low_light": "Lights off except the dim lamp on its mark (< 100 lux at the track).",
    "shadows": "Lights on. Low lamp on its mark. Assistant sweeps the cardboard shadow over the marked section as the car passes.",
    "distractors": "Lights on. All 6 distractor objects on their marks.",
    "low_battery": "Lights on. Low battery (2S LiPo 7.3-7.4 V resting; never below 7.0 V).",
    "offset_start": "Lights on. Full battery. Car centre 10 cm toward the outer edge line at the start mark, pointing straight.",
}


def find_trial(schedule_path, trial_id: str) -> dict:
    with open(schedule_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["trial_id"] == trial_id:
                return row
    raise SystemExit(f"trial {trial_id} not found in {schedule_path}")


def ask(prompt: str, parse, input_fn=input):
    while True:
        try:
            return parse(input_fn(prompt).strip())
        except ValueError:
            print("  not understood, try again")


def parse_yes_no(text: str) -> bool:
    if text.lower() in ("y", "yes"):
        return True
    if text.lower() in ("n", "no"):
        return False
    raise ValueError(text)


def append_result(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None, input_fn=input, popen=subprocess.Popen):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_id")
    parser.add_argument("--schedule", default=CAR_DIR / "schedule.csv")
    parser.add_argument("--settings", default=CAR_DIR / "settings.json")
    parser.add_argument("--data-dir", default=ROOT / "data" / "real")
    parser.add_argument("--models-dir", default=ROOT / "models")
    args = parser.parse_args(argv)

    trial_id = f"{int(args.trial_id):03d}"
    trial = find_trial(args.schedule, trial_id)
    settings = json.loads(Path(args.settings).read_text())
    policy_path = Path(args.models_dir) / f"{trial['model']}.npz"
    if not policy_path.exists():
        raise SystemExit(f"missing model file {policy_path}")

    print(f"Trial {trial_id}: condition {trial['condition']}, round {trial['round']}")
    print("  " + INSTRUCTIONS[trial["condition"]])
    battery = ask("battery voltage (V): ", float, input_fn)
    lux = ask("light at the track surface (lux): ", float, input_fn)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"trial_{trial_id}.csv"
    env = {
        **os.environ,
        "BENCH_POLICY": str(policy_path),
        "BENCH_THROTTLE": str(settings["throttle"]),
        "BENCH_LOG": str(data_dir / log_name),
        "BENCH_IMAGES": str(data_dir / f"trial_{trial_id}_img"),
        "BENCH_IMAGE_EVERY": str(settings["image_every"]),
    }
    print("Starting donkeycar. Car on the start mark, then switch to autopilot (local) in the web UI.")
    print("Wheel outside an edge line: switch to user, put the car back at that spot, switch to local.")
    print("After 3 takeovers, or at the finish mark: switch to user, then press Ctrl+C here.")
    proc = popen([sys.executable, "manage.py", "drive"], cwd=os.path.expanduser(settings["car_dir"]), env=env)
    while True:
        try:
            proc.wait()
            break
        except KeyboardInterrupt:
            # donkeycar received the same Ctrl+C; wait for it to stop its parts cleanly.
            continue

    log_path = data_dir / log_name
    if not log_path.exists():
        raise SystemExit(f"donkeycar exited without writing {log_path}; trial not recorded")

    completed = ask("lap completed (y/n): ", parse_yes_no, input_fn)
    notes = input_fn("notes: ").strip()
    append_result(data_dir / "trials.csv", {
        "trial_id": trial_id,
        "condition": trial["condition"],
        "model": trial["model"],
        "round": trial["round"],
        "battery_v": battery,
        "lux": lux,
        "completed": int(completed),
        "notes": notes,
        "log": log_name,
    })
    print(f"recorded trial {trial_id}")


if __name__ == "__main__":
    main()
