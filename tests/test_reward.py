import pytest

from envs.reward import CRASH_PENALTY, compute_reward, is_hit


def test_is_hit_treats_none_string_as_no_collision():
    assert is_hit("none") is False
    assert is_hit(None) is False
    assert is_hit("wall") is True


def test_lane_centre_gives_full_reward():
    assert compute_reward(cte=0.0, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=False) == (1.0, False)


def test_half_way_to_edge_gives_half_reward():
    reward, done = compute_reward(cte=-1.0, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=False)
    assert reward == pytest.approx(0.5)
    assert not done


def test_cte_spike_without_sim_done_is_not_terminal():
    assert compute_reward(cte=9.0, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=False) == (0.0, False)


@pytest.mark.parametrize("forward_vel", [0.0, -2.0])
def test_not_moving_forward_earns_nothing(forward_vel):
    assert compute_reward(cte=0.0, hit="none", forward_vel=forward_vel, cte_max=2.0, sim_done=False) == (0.0, False)


def test_collision_ends_episode_with_penalty():
    assert compute_reward(cte=0.0, hit="wall", forward_vel=1.0, cte_max=2.0, sim_done=False) == (CRASH_PENALTY, True)


def test_sim_done_ends_episode_with_penalty():
    assert compute_reward(cte=2.5, hit="none", forward_vel=1.0, cte_max=2.0, sim_done=True) == (CRASH_PENALTY, True)
