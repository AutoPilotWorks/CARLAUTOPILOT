import math

import pytest

from carlautopilot.core import Command
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.runner import run_episode
from carlautopilot.scenarios_lib import compose, curve, delivery_route, random_scenarios
from carlautopilot.sim import FakeEnv


def length(sc):
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(sc.route, sc.route[1:]))


def test_compose_matches_curve_geometry():
    a = compose([("straight", 60.0), ("left", 40.0, 90.0), ("straight", 60.0)])
    b = curve("left")
    assert len(a.route) == len(b.route)
    assert all(math.isclose(p.x, q.x, abs_tol=1e-6) and math.isclose(p.y, q.y, abs_tol=1e-6) for p, q in zip(a.route, b.route))


def test_compose_commands_and_heading():
    sc = compose([("straight", 100.0), ("right", 30.0, 90.0), ("straight", 50.0)])
    assert [c.command for c in sc.commands] == [Command.STRAIGHT, Command.RIGHT, Command.STRAIGHT]
    assert sc.commands[1].s_m == pytest.approx(90.0)
    end = sc.route[-1]
    assert end.x == pytest.approx(130.0, abs=0.5) and end.y == pytest.approx(-80.0, abs=0.5)
    with pytest.raises(ValueError):
        compose([("up", 10.0, 10.0)])


def test_baseline_completes_delivery_route():
    sc = delivery_route()
    assert length(sc) > 600.0
    r = run_episode(FakeEnv(privileged_obs=True), PathFollowPolicy(), sc)
    assert r.termination == "route_complete" and r.metrics["events"] == {}
    assert max(abs(x.truth.lateral_error_m) for x in r.records) < 1.5
    assert {x.command for x in r.records} == {int(Command.STRAIGHT), int(Command.LEFT), int(Command.RIGHT)}


def test_new_random_families_valid_and_complete():
    scs = random_scenarios(8, seed=3, families=("red_light", "stop_sign", "route", "curve"))
    for sc in scs:
        sc.validate()
        r = run_episode(FakeEnv(privileged_obs=True), PathFollowPolicy(), sc)
        assert r.termination == "route_complete" and r.metrics["events"] == {}, sc.name
