import math

import pytest

from carlautopilot.core import Action, Command, Env, EventKind
from carlautopilot.scenario import LeadVehicleSpec, RouteCommand, ScenarioSpec, TrafficLightSpec, Waypoint
from carlautopilot.sim import FakeEnv


def straight(length=300.0, **kw):
    base = dict(name="straight", route=(Waypoint(0.0, 0.0), Waypoint(length, 0.0)), max_time_s=120.0)
    base.update(kw)
    return ScenarioSpec(**base)


def drive(env, scenario, policy, max_steps=20000, seed=None):
    obs = env.reset(scenario, seed)
    results = []
    for _ in range(max_steps):
        res = env.step(policy(obs))
        results.append(res)
        obs = res.obs
        if res.done:
            break
    return results


def cruise(target):
    def policy(obs):
        return Action(0.0, min(1.0, max(0.0, 0.5 * (target - obs.speed_mps))), 0.0)
    return policy


def test_protocol():
    assert isinstance(FakeEnv(), Env)


def test_determinism():
    sc = straight(meta={"initial_speed_mps": 3.0})
    a = drive(FakeEnv(), sc, lambda o: Action(0.1, 0.3, 0.0), max_steps=200)
    b = drive(FakeEnv(), sc, lambda o: Action(0.1, 0.3, 0.0), max_steps=200)
    assert [r.truth for r in a] == [r.truth for r in b]


def test_noise_is_seeded():
    sc = straight(meta={"initial_speed_mps": 5.0})
    def speeds(seed):
        env = FakeEnv(speed_noise_std=0.5)
        return [r.obs.speed_mps for r in drive(env, sc, lambda o: Action(0.0, 0.2, 0.0), max_steps=30, seed=seed)]
    assert speeds(1) == speeds(1)
    assert speeds(1) != speeds(2)


def test_straight_completion():
    res = drive(FakeEnv(), straight(200.0), cruise(10.0))
    assert res[-1].termination == "route_complete"
    assert res[-1].truth.route_progress_m >= 199.0
    assert not any(r.events for r in res)


def test_steer_sign_right_positive():
    sc = straight(1000.0, meta={"initial_speed_mps": 5.0})
    right = drive(FakeEnv(), sc, lambda o: Action(0.5, 0.0, 0.0), max_steps=20)
    assert right[-1].truth.yaw_rad < 0 and right[-1].truth.y < 0 and right[-1].truth.lateral_error_m < 0
    left = drive(FakeEnv(), sc, lambda o: Action(-0.5, 0.0, 0.0), max_steps=20)
    assert left[-1].truth.yaw_rad > 0 and left[-1].truth.y > 0 and left[-1].truth.lateral_error_m > 0


def test_speed_never_negative():
    res = drive(FakeEnv(), straight(meta={"initial_speed_mps": 2.0}), lambda o: Action(0.0, 0.0, 1.0), max_steps=100)
    assert all(r.truth.speed_mps >= 0.0 for r in res)


def test_lead_collision():
    sc = straight(lead=LeadVehicleSpec(20.0, ((0.0, 0.0),)), meta={"initial_speed_mps": 10.0})
    res = drive(FakeEnv(), sc, lambda o: Action(0.0, 0.0, 0.0))
    assert res[-1].termination == "collision"
    assert [e.kind for r in res for e in r.events] == [EventKind.COLLISION_VEHICLE]


def test_lead_moving_away_is_not_collision():
    sc = straight(200.0, lead=LeadVehicleSpec(20.0, ((0.0, 15.0),)))
    res = drive(FakeEnv(), sc, cruise(8.0))
    assert res[-1].termination == "route_complete"


def test_red_light_one_event_per_light():
    lights = (TrafficLightSpec(50.0, red_s=100.0, green_s=1.0), TrafficLightSpec(100.0, red_s=100.0, green_s=1.0))
    sc = straight(200.0, lights=lights, meta={"initial_speed_mps": 10.0})
    res = drive(FakeEnv(), sc, cruise(10.0))
    kinds = [e.kind for r in res for e in r.events]
    assert kinds == [EventKind.RED_LIGHT, EventKind.RED_LIGHT]


def test_green_light_no_event():
    sc = straight(200.0, lights=(TrafficLightSpec(50.0, red_s=1.0, green_s=1000.0, offset_s=1.0),), meta={"initial_speed_mps": 10.0})
    res = drive(FakeEnv(), sc, cruise(10.0))
    assert not any(r.events for r in res)


def test_stopping_before_red_gives_no_light_event():
    sc = straight(200.0, lights=(TrafficLightSpec(60.0, red_s=1000.0, green_s=1.0),), meta={"initial_speed_mps": 10.0}, max_time_s=200.0)

    env = FakeEnv()
    env.reset(sc)
    events = []
    for _ in range(4000):
        s = env.ground_truth().route_progress_m
        res = env.step(Action(0.0, 0.0, 1.0 if s > 30.0 else 0.0))
        events += list(res.events)
        if res.done:
            break
    assert res.truth.route_progress_m < 60.0
    assert EventKind.RED_LIGHT not in [e.kind for e in events]
    assert res.termination == "blocked"


def test_off_road():
    sc = straight(1000.0, meta={"initial_speed_mps": 5.0})
    res = drive(FakeEnv(), sc, lambda o: Action(1.0, 0.1, 0.0))
    assert res[-1].termination == "off_route"
    assert EventKind.OFF_ROAD in [e.kind for r in res for e in r.events]


def test_timeout():
    sc = straight(1000.0, max_time_s=5.0, meta={"initial_speed_mps": 1.0})
    res = drive(FakeEnv(), sc, lambda o: Action(0.0, 0.05, 0.0))
    assert res[-1].termination == "timeout"
    assert res[-1].obs.t >= 5.0


def test_blocked():
    sc = straight(1000.0, max_time_s=120.0)
    res = drive(FakeEnv(), sc, lambda o: Action(0.0, 0.0, 0.0))
    assert res[-1].termination == "blocked"
    assert 29.9 <= res[-1].obs.t <= 30.2
    assert EventKind.BLOCKED in [e.kind for e in res[-1].events]


def test_lead_visibility_limit():
    sc = straight(500.0, lead=LeadVehicleSpec(150.0, ((0.0, 0.0),)), meta={"initial_speed_mps": 20.0})
    env = FakeEnv()
    obs = env.reset(sc)
    assert obs.lead is None
    first = None
    for _ in range(2000):
        res = env.step(Action(0.0, 0.1, 0.0))
        if res.obs.lead is not None:
            first = res.obs.lead
            break
        if res.done:
            break
    assert first is not None and 90.0 < first.distance_m <= 100.0
    assert first.rel_speed_mps < 0


def test_command_follows_progress():
    cmds = (RouteCommand(10.0, Command.LEFT), RouteCommand(50.0, Command.RIGHT))
    sc = straight(100.0, commands=cmds, meta={"initial_speed_mps": 10.0})
    res = drive(FakeEnv(), sc, cruise(10.0))
    seen = []
    for r in res:
        c = r.obs.command
        if not seen or seen[-1] != c:
            seen.append(c)
        if c == Command.LEFT:
            assert 9.0 <= r.truth.route_progress_m < 51.0
    assert seen == [Command.FOLLOW, Command.LEFT, Command.RIGHT] or seen == [Command.LEFT, Command.RIGHT]


def test_privileged_obs():
    sc = straight(200.0, lights=(TrafficLightSpec(60.0, red_s=100.0, green_s=1.0),), meta={"initial_speed_mps": 5.0})
    assert FakeEnv().reset(sc).extras == {"route_target": (30.0, 0.0)}  # navigation only: the route 30 m ahead
    env = FakeEnv(privileged_obs=True)
    obs = env.reset(sc)
    ex = obs.extras
    assert set(ex) >= {"heading_error_rad", "lateral_error_m", "lookahead_points", "next_light_distance_m", "next_light_red"}
    assert abs(ex["heading_error_rad"]) < 1e-9 and abs(ex["lateral_error_m"]) < 1e-9
    assert ex["next_light_red"] is True and ex["next_light_distance_m"] == pytest.approx(60.0)
    for (x, y), d in zip(ex["lookahead_points"], (3.0, 6.0, 9.0)):
        assert x == pytest.approx(d) and y == pytest.approx(0.0, abs=1e-9)


def test_lookahead_frame_on_left_curve():
    sc = ScenarioSpec(name="l", route=(Waypoint(0.0, 0.0), Waypoint(10.0, 0.0), Waypoint(10.0, 100.0)))
    env = FakeEnv(privileged_obs=True)
    env.reset(sc)
    pts = env.reset(sc).extras["lookahead_points"]
    assert any(y > 1.0 for _, y in pts)


def test_reset_twice_and_misuse():
    env = FakeEnv()
    with pytest.raises(RuntimeError):
        env.step(Action())
    sc = straight(meta={"initial_speed_mps": 2.0})
    a = env.reset(sc)
    env.step(Action(0.0, 1.0, 0.0))
    b = env.reset(sc)
    assert a.speed_mps == b.speed_mps == 2.0 and b.t == 0.0
    res = drive(env, straight(30.0, meta={"initial_speed_mps": 10.0}), cruise(10.0))
    assert res[-1].done
    with pytest.raises(RuntimeError):
        env.step(Action())


def test_degenerate_route_rejected():
    sc = ScenarioSpec(name="d", route=(Waypoint(0, 0), Waypoint(0, 0), Waypoint(5, 0)))
    with pytest.raises(ValueError):
        FakeEnv().reset(sc)


def test_speed_response_matches_model():
    env = FakeEnv(drag=0.0)
    env.reset(straight())
    res = [env.step(Action(0.0, 1.0, 0.0)) for _ in range(20)]
    assert res[-1].truth.speed_mps == pytest.approx(3.0 * 20 * env.dt)
    assert math.isfinite(res[-1].truth.x)
