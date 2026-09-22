import socket
import sys
import time

import pytest

from envs.sim_process import SimDisconnectedError, call_with_timeout, kill_sim, launch_sim


def test_call_with_timeout_returns_value():
    assert call_with_timeout(lambda x: x * 2, 1.0, 21) == 42


def test_call_with_timeout_raises_on_hang():
    start = time.monotonic()
    with pytest.raises(SimDisconnectedError):
        call_with_timeout(time.sleep, 0.2, 5)
    assert time.monotonic() - start < 2


def test_call_with_timeout_reraises_errors():
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        call_with_timeout(boom, 1.0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fake_sim(tmp_path, body: str):
    """Write an executable script that accepts the real sim's CLI args."""
    script = tmp_path / "fake_sim"
    script.write_text(
        f"#!{sys.executable}\n"
        "import socket, sys, time\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        f"{body}\n"
    )
    script.chmod(0o755)
    return str(script)


def test_launch_sim_waits_for_port(tmp_path):
    exe = _fake_sim(
        tmp_path,
        "time.sleep(0.5)\n"
        "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "s.bind(('127.0.0.1', port)); s.listen()\n"
        "time.sleep(60)",
    )
    proc = launch_sim(exe, _free_port(), timeout=10)
    try:
        assert proc.poll() is None
    finally:
        kill_sim(proc)
    assert proc.poll() is not None


def test_launch_sim_reports_early_exit(tmp_path):
    exe = _fake_sim(tmp_path, "sys.exit(3)")
    with pytest.raises(SimDisconnectedError, match="code 3"):
        launch_sim(exe, _free_port(), timeout=10)


def test_launch_sim_times_out_and_kills(tmp_path):
    exe = _fake_sim(tmp_path, "time.sleep(60)")
    with pytest.raises(SimDisconnectedError, match="did not open port"):
        launch_sim(exe, _free_port(), timeout=1.5)


def test_kill_sim_accepts_none():
    kill_sim(None)
