import argparse

import pytest

from envs.cli import add_env_args, env_kwargs_from_args


def _parse(argv, monkeypatch, env_var=None):
    if env_var is None:
        monkeypatch.delenv("DONKEY_SIM_PATH", raising=False)
    else:
        monkeypatch.setenv("DONKEY_SIM_PATH", env_var)
    parser = argparse.ArgumentParser()
    add_env_args(parser)
    return parser.parse_args(argv)


def test_exe_path_falls_back_to_env_var(monkeypatch):
    assert env_kwargs_from_args(_parse([], monkeypatch, "/sims/donkey"))["exe_path"] == "/sims/donkey"


def test_missing_exe_path_exits_with_message(monkeypatch):
    with pytest.raises(SystemExit, match="DONKEY_SIM_PATH"):
        env_kwargs_from_args(_parse([], monkeypatch))


def test_flags_reach_kwargs(monkeypatch):
    args = _parse(["--exe-path", "remote", "--throttle", "0.3", "--cte-max", "1.5", "--cam-fov", "49",
                   "--cte-offset", "-6.6"], monkeypatch)
    kwargs = env_kwargs_from_args(args)
    assert kwargs["throttle"] == 0.3
    assert kwargs["cte_max"] == 1.5
    assert kwargs["cam_fov"] == 49
    assert kwargs["cte_offset"] == -6.6
