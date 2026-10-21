import pytest

from carlautopilot.core import Action, Command, GroundTruth, Observation, StepResult
from carlautopilot.scenario import LeadVehicleSpec, RouteCommand, ScenarioSpec, TrafficLightSpec, Waypoint


def make_spec(**kw):
    base = dict(
        name="s1",
        route=(Waypoint(0, 0), Waypoint(100, 0), Waypoint(100, 50)),
        commands=(RouteCommand(10.0, Command.STRAIGHT), RouteCommand(100.0, Command.LEFT)),
        lead=LeadVehicleSpec(30.0, ((0.0, 8.0), (20.0, 4.0))),
        lights=(TrafficLightSpec(60.0, 8.0, 12.0, 1.0),),
        tags=("smoke",),
        meta={"k": 1},
    )
    base.update(kw)
    return ScenarioSpec(**base)


def test_roundtrip(tmp_path):
    spec = make_spec()
    p = tmp_path / "s.json"
    spec.save(p)
    assert ScenarioSpec.load(p) == spec


@pytest.mark.parametrize(
    "kw",
    [
        dict(route=(Waypoint(0, 0),)),
        dict(max_time_s=0.0),
        dict(commands=(RouteCommand(5.0, Command.LEFT), RouteCommand(1.0, Command.RIGHT))),
        dict(lead=LeadVehicleSpec(-1.0)),
        dict(lead=LeadVehicleSpec(10.0, ())),
        dict(lights=(TrafficLightSpec(5.0, red_s=0.0),)),
    ],
)
def test_validation_rejects(kw):
    with pytest.raises(ValueError):
        make_spec(**kw).validate()


def test_action_clipped():
    a = Action(steer=3.0, throttle=-1.0, brake=2.0).clipped()
    assert (a.steer, a.throttle, a.brake) == (1.0, 0.0, 1.0)


def test_step_result_defaults():
    obs = Observation(t=0.0, speed_mps=0.0, command=Command.FOLLOW)
    truth = GroundTruth(0, 0, 0, 0, 0, 100, 0)
    r = StepResult(obs=obs, truth=truth, done=False)
    assert r.events == () and r.termination is None
