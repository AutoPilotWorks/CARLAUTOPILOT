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


def test_model_policy_caps_speed_at_junction_commands():
    import numpy as np

    from carlautopilot.core import Command, Observation
    from carlautopilot.models.route_steering import RouteSteeringNet
    from carlautopilot.policies.model_policy import ModelPolicy

    img = np.zeros((32, 64, 3), dtype=np.uint8)
    pol = ModelPolicy(RouteSteeringNet(), speed_limit_mps=8.0, turn_speed_mps=4.5)
    fast = dict(t=0.0, speed_mps=7.0, image=img)
    assert pol.act(Observation(command=Command.FOLLOW, **fast)).brake == 0.0
    turning = pol.act(Observation(command=Command.LEFT, **fast))
    assert turning.brake > 0.0 and turning.throttle == 0.0


def test_learned_speed_replaces_the_junction_rule():
    import numpy as np
    import torch

    from carlautopilot.core import Command, Observation
    from carlautopilot.models.route_steering import RouteSteeringNet
    from carlautopilot.policies.model_policy import ModelPolicy

    net = RouteSteeringNet(predict_speed=True)
    with torch.no_grad():
        for head in net.heads:
            head[-1].weight.zero_()
            head[-1].bias[3] = 3.0 if head is net.heads[Command.FOLLOW] else 9.0
    pol = ModelPolicy(net, speed_limit_mps=8.0, turn_speed_mps=4.5)
    obs = dict(t=0.0, speed_mps=7.0, image=np.zeros((32, 64, 3), dtype=np.uint8))
    assert pol.act(Observation(command=Command.FOLLOW, **obs)).brake > 0.0
    assert pol.act(Observation(command=Command.LEFT, **obs)).brake == 0.0


def test_perceiving_model_uses_its_own_red_light_and_lead_not_the_simulators():
    import numpy as np
    import torch

    from carlautopilot.core import Command, LeadInfo, Observation
    from carlautopilot.models.route_steering import RouteSteeringNet
    from carlautopilot.policies.model_policy import ModelPolicy

    def policy(red_logit, lead_gap):
        net = RouteSteeringNet(predict_speed=True, perceive=True)
        with torch.no_grad():
            for head in net.heads:
                head[-1].weight.zero_()
                head[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 9.0, red_logit, 6.0, lead_gap, -2.0]))
        return ModelPolicy(net, speed_limit_mps=8.0)

    img = np.zeros((32, 64, 3), dtype=np.uint8)
    sim_lead = LeadInfo(distance_m=3.0, rel_speed_mps=-5.0)
    obs = Observation(t=0.0, speed_mps=6.0, command=Command.FOLLOW, image=img, lead=sim_lead)
    clear = policy(-9.0, 60.0).act(obs)
    assert clear.brake == 0.0 and clear.throttle > 0.0
    assert policy(9.0, 60.0).act(obs).brake > 0.0
    assert policy(-9.0, 8.0).act(obs).brake > 0.0


def test_expert_stops_for_a_yield_line():
    from carlautopilot.core import Command, Observation
    from carlautopilot.policies import PathFollowPolicy

    pts = [(d, 0.0) for d in (3.0, 6.0, 9.0, 12.0)]
    p = PathFollowPolicy(speed_limit_mps=8.0)
    base = dict(t=0.0, speed_mps=6.0, command=Command.LEFT)
    assert p.act(Observation(**base, extras={"lookahead_points": pts})).brake == 0.0
    assert p.act(Observation(**base, extras={"lookahead_points": pts, "yield_distance_m": 8.0})).brake > 0.0
