import math

import pytest

from carlautopilot.policies import PathFollowPolicy


def test_speed_cap_slows_for_tight_turns_only_when_enabled():
    straight = [(d, 0.0) for d in (5.0, 10.0, 20.0)]
    r = 10.0
    turn = [(r * math.sin(a), -r * (1 - math.cos(a))) for a in (0.3, 0.6, 0.9)]
    p = PathFollowPolicy(speed_limit_mps=8.0, max_lat_accel=2.0)
    assert p.speed_cap(straight) == 8.0
    assert p.speed_cap(turn) == pytest.approx(math.sqrt(2.0 * r), rel=1e-6)
    assert PathFollowPolicy(speed_limit_mps=8.0).speed_cap(turn) == 8.0
    assert p.speed_cap([(-1.0, 0.0)]) == 8.0
