import socket
import subprocess
import threading
import time


class SimDisconnectedError(RuntimeError):
    """The simulator failed to start or stopped answering.

    Recovery is always the same: close the env and build a new one.
    """


def call_with_timeout(fn, timeout: float, *args):
    """Run fn(*args) in a daemon thread. Raise SimDisconnectedError if it
    hasn't returned after timeout seconds; re-raise anything fn raised.

    gym-donkeycar busy-waits forever for the next frame when the sim dies,
    so a timeout is the only way to notice. The stuck thread is a daemon,
    so it can't keep the process alive at exit.
    """
    result = {}

    def target():
        try:
            result["value"] = fn(*args)
        except BaseException as exc:  # handed back to the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise SimDisconnectedError(f"simulator did not respond within {timeout}s")
    if "error" in result:
        raise result["error"]
    return result["value"]


def launch_sim(exe_path: str, port: int, timeout: float) -> subprocess.Popen:
    """Start the simulator binary and wait until it accepts TCP connections.

    We own the process (rather than letting gym-donkeycar start it) so a
    failed start never leaves an orphaned sim holding the port.
    """
    proc = subprocess.Popen(
        [exe_path, "--port", str(port), "--host", "127.0.0.1", "-logFile", f"unitylog_{port}.txt"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SimDisconnectedError(f"simulator exited during startup with code {proc.returncode}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
            return proc
        except OSError:
            time.sleep(0.5)
    kill_sim(proc)
    raise SimDisconnectedError(f"simulator did not open port {port} within {timeout}s")


def kill_sim(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.kill()
    proc.wait(timeout=10)
