import csv
import json

import pytest

from car.trial import main


def _setup(tmp_path):
    schedule = tmp_path / "schedule.csv"
    with schedule.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "condition", "model", "round"])
        writer.writerow(["007", "shadows", "bc_dr", "2"])
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"throttle": 0.3, "car_dir": str(tmp_path / "mycar"), "image_every": 5}))
    models = tmp_path / "models"
    models.mkdir()
    (models / "bc_dr.npz").touch()
    data = tmp_path / "real"
    argv = ["7", "--schedule", str(schedule), "--settings", str(settings), "--data-dir", str(data), "--models-dir", str(models)]
    return argv, data


class FakePopen:
    instances = []

    def __init__(self, cmd, cwd, env):
        self.cmd, self.cwd, self.env = cmd, cwd, env
        self.waits = 0
        FakePopen.instances.append(self)

    def wait(self):
        self.waits += 1
        if self.waits == 1:
            raise KeyboardInterrupt  # operator's Ctrl+C; donkeycar is still shutting down
        return 0


def test_trial_runs_donkeycar_and_records_result(tmp_path, capsys):
    argv, data = _setup(tmp_path)
    answers = iter(["abc", "8.2", "350", "maybe", "y", "slight wobble"])
    main(argv, input_fn=lambda prompt: next(answers), popen=FakePopen)

    proc = FakePopen.instances[-1]
    assert proc.cmd[-2:] == ["manage.py", "drive"]
    assert proc.cwd == str(tmp_path / "mycar")
    assert proc.env["BENCH_POLICY"].endswith("bc_dr.npz")
    assert proc.env["BENCH_THROTTLE"] == "0.3"
    assert proc.env["BENCH_LOG"].endswith("trial_007.csv")
    assert proc.waits == 2  # kept waiting for a clean donkeycar shutdown after Ctrl+C

    with (data / "trials.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert (row["trial_id"], row["condition"], row["model"], row["round"]) == ("007", "shadows", "bc_dr", "2")
    assert float(row["battery_v"]) == 8.2 and float(row["lux"]) == 350.0
    assert row["completed"] == "1" and row["notes"] == "slight wobble" and row["log"] == "trial_007.csv"
    assert "bc_dr" not in capsys.readouterr().out  # operator stays blind to the model


def test_unknown_trial_exits(tmp_path):
    argv, _ = _setup(tmp_path)
    argv[0] = "99"
    with pytest.raises(SystemExit, match="099"):
        main(argv, input_fn=lambda prompt: "1", popen=FakePopen)


def test_missing_model_file_exits_before_driving(tmp_path):
    argv, _ = _setup(tmp_path)
    (tmp_path / "models" / "bc_dr.npz").unlink()
    count = len(FakePopen.instances)
    with pytest.raises(SystemExit, match="bc_dr.npz"):
        main(argv, input_fn=lambda prompt: "1", popen=FakePopen)
    assert len(FakePopen.instances) == count
