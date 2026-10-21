import json

import pytest

from carlautopilot.core import Action, EventKind, Policy
from carlautopilot.cli.evaluate import main as evaluate_main
from carlautopilot.eval import read_episode
from carlautopilot.policies import ConstantPolicy, PathFollowPolicy
from carlautopilot.runner import run_episode, run_suite
from carlautopilot.scenarios_lib import curve, lead_brake, red_light, standard_suite, straight
from carlautopilot.sim import FakeEnv


def env():
    return FakeEnv(privileged_obs=True)


def test_policy_protocol():
    assert isinstance(PathFollowPolicy(), Policy) and isinstance(ConstantPolicy(), Policy)


def test_straight_completes_with_full_score():
    r = run_episode(env(), PathFollowPolicy(), straight())
    assert r.termination == "route_complete"
    assert r.metrics["driving_score"] == pytest.approx(1.0) and r.metrics["success"]
    assert r.metrics["events"] == {}


@pytest.mark.parametrize("direction", ["left", "right"])
def test_curves_completed_without_events(direction):
    r = run_episode(env(), PathFollowPolicy(speed_limit_mps=6.0), curve(direction))
    assert r.termination == "route_complete", r.metrics
    assert r.metrics["events"] == {}
    assert max(abs(x.truth.lateral_error_m) for x in r.records) < 1.5


def test_curve_steering_sign():
    left = run_episode(env(), PathFollowPolicy(speed_limit_mps=6.0), curve("left"))
    right = run_episode(env(), PathFollowPolicy(speed_limit_mps=6.0), curve("right"))
    assert min(x.action.steer for x in left.records) < -0.1
    assert max(x.action.steer for x in right.records) > 0.1


def test_lead_braking_no_collision():
    r = run_episode(env(), PathFollowPolicy(), lead_brake())
    assert r.termination == "route_complete"
    assert not any(k.startswith("collision") for k in r.metrics["events"])
    assert r.metrics["min_lead_distance_m"] > 1.0


def test_unsafe_baseline_collides_with_lead():
    r = run_episode(env(), ConstantPolicy(Action(0.0, 0.6, 0.0)), lead_brake())
    assert r.termination == "collision" and not r.metrics["success"]
    assert r.metrics["driving_score"] < 1.0


class Cruise:
    def __init__(self, safe):
        self.safe = safe
        self.name = "cruise_safe" if safe else "cruise_unsafe"

    def reset(self):
        pass

    def act(self, obs):
        from carlautopilot.control import apply_safety

        action = Action(0.0, min(1.0, max(0.0, 0.5 * (12.0 - obs.speed_mps))), 0.0)
        return apply_safety(action, obs) if self.safe else action


def test_safety_layer_alone_prevents_the_collision():
    unsafe = run_episode(env(), Cruise(safe=False), lead_brake())
    safe = run_episode(env(), Cruise(safe=True), lead_brake())
    assert unsafe.termination == "collision"
    assert not any(k.startswith("collision") for k in safe.metrics["events"])
    assert safe.metrics["min_lead_distance_m"] > 0.0


def test_stops_for_red_light_then_proceeds():
    r = run_episode(env(), PathFollowPolicy(), red_light())
    assert r.termination == "route_complete"
    assert EventKind.RED_LIGHT not in r.metrics["events"]
    assert min(x.speed_mps for x in r.records if 60.0 < x.truth.route_progress_m < 125.0) < 0.5


def test_running_the_light_is_penalized():
    r = run_episode(env(), ConstantPolicy(Action(0.0, 0.5, 0.0)), red_light())
    assert r.metrics["events"].get(EventKind.RED_LIGHT) == 1
    assert r.metrics["driving_score"] <= 0.7 + 1e-9


def test_episode_log_roundtrip_matches_result(tmp_path):
    p = tmp_path / "ep.jsonl"
    r = run_episode(env(), PathFollowPolicy(), straight(100.0), log_path=p)
    header, records = read_episode(p)
    assert header == r.header and records == r.records


def test_repeatability():
    a = run_episode(env(), PathFollowPolicy(), lead_brake())
    b = run_episode(env(), PathFollowPolicy(), lead_brake())
    assert a.records == b.records and a.metrics == b.metrics


def test_run_suite_outputs(tmp_path):
    res = run_suite(env, PathFollowPolicy, standard_suite(), tmp_path / "out")
    agg = res["aggregate"]
    assert agg["episodes"] == 7 and agg["success_rate"] == 1.0 and agg["mean_driving_score"] == pytest.approx(1.0)
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8").startswith("# Evaluation report")
    assert len(list((tmp_path / "out" / "logs").glob("*.jsonl"))) == 7
    with pytest.raises(FileExistsError):
        run_suite(env, PathFollowPolicy, standard_suite(), tmp_path / "out")


def test_cli_with_json_scenarios(tmp_path):
    d = tmp_path / "sc"
    d.mkdir()
    straight(80.0, name="s80").save(d / "s80.json")
    red_light(name="rl").save(d / "rl.json")
    res = evaluate_main(["--out-dir", str(tmp_path / "res"), "--scenarios-dir", str(d)])
    assert res["aggregate"]["episodes"] == 2
    data = json.loads((tmp_path / "res" / "metrics.json").read_text(encoding="utf-8"))
    assert {e["scenario"] for e in data["episodes"]} == {"s80", "rl"}


def test_scenarios_are_valid_and_roundtrip(tmp_path):
    for sc in standard_suite():
        sc.validate()
        sc.save(tmp_path / f"{sc.name}.json")
        assert type(sc).load(tmp_path / f"{sc.name}.json") == sc
    with pytest.raises(ValueError):
        curve("up")
