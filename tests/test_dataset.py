import numpy as np
import pytest

from car.policy import CAMERA_SHAPE
from collect.dataset import DatasetWriter, load_frame, next_frame_index, read_labels


def _frame(value):
    return np.full(CAMERA_SHAPE, value, dtype=np.uint8)


def test_writer_flushes_in_batches_and_reads_back(tmp_path):
    writer = DatasetWriter(tmp_path, save_every=3)
    for i in range(7):
        writer.add(_frame(i), steer=i / 10, prev_steer=-i / 10, episode="run-0", source="scripted")
    assert len(read_labels(tmp_path)) == 6
    assert writer.total == 7
    writer.flush()
    rows = read_labels(tmp_path)
    assert len(rows) == 7
    assert rows[3]["steer"] == pytest.approx(0.3)
    assert rows[3]["prev_steer"] == pytest.approx(-0.3)
    assert rows[3]["episode"] == "run-0" and rows[3]["source"] == "scripted"
    assert np.array_equal(load_frame(rows[3]["frame"]), _frame(3))


def test_second_writer_appends(tmp_path):
    for value in (1, 2):
        writer = DatasetWriter(tmp_path)
        writer.add(_frame(value), 0.0, 0.0, f"run-{value}", "manual")
        writer.flush()
    rows = read_labels(tmp_path)
    assert [r["episode"] for r in rows] == ["run-1", "run-2"]
    assert np.array_equal(load_frame(rows[1]["frame"]), _frame(2))


def test_deleting_a_frame_never_causes_overwrite(tmp_path):
    writer = DatasetWriter(tmp_path)
    for _ in range(3):
        writer.add(_frame(0), 0.0, 0.0, "e", "scripted")
    writer.flush()
    (tmp_path / "frame_00000000.png").unlink()
    assert next_frame_index(tmp_path) == 3


def test_rgb_order_survives_roundtrip(tmp_path):
    frame = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    frame[..., 0] = 200
    writer = DatasetWriter(tmp_path)
    writer.add(frame, 0.0, 0.0, "e", "scripted")
    writer.flush()
    assert np.array_equal(load_frame(read_labels(tmp_path)[0]["frame"]), frame)


def test_read_labels_of_missing_dir_is_empty(tmp_path):
    assert read_labels(tmp_path / "nothing") == []
