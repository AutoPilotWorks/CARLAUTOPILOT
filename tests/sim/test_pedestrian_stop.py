import dataclasses

import pytest

from carlautopilot.core import Action, EventKind
from carlautopilot.policies import ConstantPolicy, PathFollowPolicy
from carlautopilot.runner import run_episode
from carlautopilot.scenario import PedestrianSpec, ScenarioSpec
from carlautopilot.scenarios_lib import pedestrian_crossing, stop_sign, straight
from carlautopilot.sim import FakeEnv


def test_spec_roundtrip_and_validation(tmp_path):
    sc = dataclasses.replace(stop_sign(), pedestrians=pedestrian_crossing().pedestrians)
    sc.save(tmp_path / "s.json")
    assert ScenarioSpec.load(tmp_path / "s.json") == sc
    for bad in (PedestrianSpec(10, 0.0), PedestrianSpec(10, 3.0, speed_mps=0.0), PedestrianSpec(10, 3.0, start_t_s=-1.0)):
        with pytest.raises(ValueError):
            dataclasses.replace(straight(), pedestrians=(bad,)).validate()


def test_pedestrian_is_obstacle_only_inside_corridor():
    sc = dataclasses.replace(straight(), pedestrians=(PedestrianSpec(40.0, 3.5, 1.4, 0.0),))
    env = FakeEnv()
    obs = env.reset(sc)
    assert obs.lead is None
    seen = []
    for _ in range(200):
        res = env.step(Action(0.0, 0.0, 1.0))
        seen.append(res.obs.lead is not None)
    assert not seen[0] and any(seen) and not seen[-1]
    assert env._ped_lateral(0, 0.0) == 3.5 and env._ped_lateral(0, 100.0) == -3.5


def test_ignoring_the_pedestrian_collides():
    r = run_episode(FakeEnv(), ConstantPolicy(Action(0.0, 0.3, 0.0)), pedestrian_crossing())
    assert r.termination == "collision"
    assert r.metrics["events"] == {EventKind.COLLISION_PEDESTRIAN: 1}
    assert not r.metrics["success"] and r.metrics["driving_score"] < 0.5


def test_baseline_yields_to_pedestrian():
    r = run_episode(FakeEnv(privileged_obs=True), PathFollowPolicy(), pedestrian_crossing())
    assert r.termination == "route_complete" and r.metrics["events"] == {}
    assert r.metrics["min_lead_distance_m"] > 1.0


def test_stop_sign_violation_is_penalised_once():
    r = run_episode(FakeEnv(), ConstantPolicy(Action(0.0, 0.3, 0.0)), stop_sign())
    assert r.metrics["events"] == {EventKind.STOP_SIGN: 1}
    assert r.metrics["driving_score"] == pytest.approx(0.8)


def test_baseline_stops_at_sign_then_continues():
    sc = stop_sign()
    r = run_episode(FakeEnv(privileged_obs=True), PathFollowPolicy(), sc)
    assert r.termination == "route_complete" and r.metrics["events"] == {}
    sign_s = sc.stop_signs[0].s_m
    near = [x for x in r.records if sign_s - 8.0 <= x.truth.route_progress_m <= sign_s]
    assert min(x.speed_mps for x in near) <= 0.5
    assert r.records[-1].t < 60.0


def test_sign_hidden_from_observation_after_clearing():
    env = FakeEnv(privileged_obs=True)
    obs = env.reset(dataclasses.replace(stop_sign(sign_s_m=90.0), meta={"initial_speed_mps": 0.0}))
    assert obs.extras["next_stop_sign_distance_m"] == pytest.approx(90.0)
    env.x = 85.0
    env.v = 0.0
    res = env.step(Action(0.0, 0.0, 1.0))
    assert "next_stop_sign_distance_m" not in res.obs.extras and res.events == ()
