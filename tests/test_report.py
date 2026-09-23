import csv
import json
import math

import pytest

from bench.report import cell_stats, compare, load_real, load_sim, sim_real_spearman, write_report


def _rows(model, condition, successes, n, jerk=1.0):
    return [{"model": model, "condition": condition, "success": i < successes, "jerk": jerk + i, "osc_hz": math.nan}
            for i in range(n)]


def test_cell_stats_counts_rates_and_ignores_nan():
    cells = cell_stats(_rows("sac_clean", "nominal", 3, 4), metrics=("jerk", "osc_hz"))
    cell = cells[("sac_clean", "nominal")]
    assert (cell["n"], cell["successes"], cell["rate"]) == (4, 3, 0.75)
    assert cell["ci"][0] < 0.75 < cell["ci"][1]
    assert cell["jerk"][0] == pytest.approx(2.5)
    assert all(math.isnan(v) for v in cell["osc_hz"])


def test_compare_pools_noise_conditions_only():
    rows = (_rows("sac_dr", "nominal", 0, 5) + _rows("sac_dr", "low_light", 5, 5) + _rows("sac_dr", "shadows", 5, 5)
            + _rows("bc_dr", "nominal", 5, 5) + _rows("bc_dr", "low_light", 0, 5) + _rows("bc_dr", "shadows", 1, 5, jerk=10))
    result = compare(rows, "sac_dr", "bc_dr")
    assert (result["n_a"], result["n_b"]) == (10, 10)
    assert result["rate_a"] == 1.0 and result["rate_b"] == 0.1
    assert 0.0 <= result["p_success"] < 0.01
    assert 0.0 <= result["p_jerk"] <= 1.0


def test_sim_real_spearman():
    sim = {("m", c): {"rate": r} for c, r in [("a", 0.1), ("b", 0.5), ("c", 0.9)]}
    real = {("m", c): {"rate": r} for c, r in [("a", 0.0), ("b", 0.2), ("c", 0.6)]}
    rho, p, n = sim_real_spearman(sim, real)
    assert rho == pytest.approx(1.0) and n == 3
    assert math.isnan(sim_real_spearman({("m", "a"): {"rate": 1}}, {("m", "a"): {"rate": 1}})[0])


def test_load_sim_and_missing_files(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"model": "bc_dr", "condition": "nominal", "episode": 0, "outcome": "crash"}) + "\n")
    assert load_sim(path)[0]["success"] is False
    assert load_sim(tmp_path / "none.jsonl") == []
    assert load_real(tmp_path / "none.csv") == []


def test_load_sim_skips_truncated_trailing_line(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        json.dumps({"model": "bc_dr", "condition": "nominal", "episode": 0, "outcome": "crash"}) + "\n"
        + '{"model": "bc_dr", "condition": "nominal", "epis'  # killed mid-write
    )
    rows = load_sim(path)
    assert len(rows) == 1 and rows[0]["success"] is False


def test_load_real_uses_log_and_answer(tmp_path):
    with (tmp_path / "trial_001.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "mode", "pilot_steer", "user_steer", "latency_ms"])
        for i in range(40):
            w.writerow([f"{i * 0.05:.3f}", "user" if i < 5 or i >= 35 else "local", "0.1000", "0.0000", "9.00"])
    with (tmp_path / "trials.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "condition", "model", "round", "battery_v", "lux", "completed", "notes", "log"])
        w.writerow(["001", "nominal", "sac_dr", "1", "8.3", "400", "1", "", "trial_001.csv"])
    [row] = load_real(tmp_path / "trials.csv")
    assert row["success"] and row["interventions"] == 0
    assert row["lap_time"] == pytest.approx(1.5)
    assert row["latency_ms_mean"] == pytest.approx(9.0)
    assert row["latency_ms_p95"] == pytest.approx(9.0)


def test_load_real_keeps_only_last_row_for_a_rerun_trial(tmp_path):
    with (tmp_path / "trial_001.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "mode", "pilot_steer", "user_steer", "latency_ms"])
        for i in range(40):
            w.writerow([f"{i * 0.05:.3f}", "user" if i < 5 or i >= 35 else "local", "0.1000", "0.0000", "9.00"])
    with (tmp_path / "trials.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "condition", "model", "round", "battery_v", "lux", "completed", "notes", "log"])
        w.writerow(["001", "nominal", "sac_dr", "1", "8.3", "400", "0", "sim crashed, rerunning", "trial_001.csv"])
        w.writerow(["001", "nominal", "sac_dr", "1", "8.1", "410", "1", "", "trial_001.csv"])
    rows = load_real(tmp_path / "trials.csv")
    assert len(rows) == 1 and rows[0]["success"] is True  # only the rerun's row counts


def test_write_report_with_sim_only_and_with_both(tmp_path):
    sim = _rows("sac_clean", "nominal", 8, 10) + _rows("sac_clean", "low_light", 4, 10) + _rows("bc_clean", "low_light", 2, 10)
    for r in sim:
        r.update(avg_abs_cte=0.3, recovery_steps=None)
    out = write_report(sim, [], tmp_path / "sim_only")
    text = out.read_text()
    assert "Simulator" in text and "sac_clean" in text
    real = [dict(r, interventions=0, latency_ms_mean=10.0) for r in sim]
    write_report(sim, real, tmp_path / "both")
    assert (tmp_path / "both" / "real_success.png").exists()
    assert (tmp_path / "both" / "sim_vs_real.png").exists()
