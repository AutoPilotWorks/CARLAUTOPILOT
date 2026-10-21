from __future__ import annotations

from dataclasses import dataclass, field

from carlautopilot.control import ACCController, PurePursuit, SafetyConfig, apply_safety
from carlautopilot.core import Action, Observation

LIGHT_STOP_MARGIN_M = 3.0
LIGHT_REACT_DISTANCE_M = 80.0


@dataclass
class ConstantPolicy:
    action: Action = field(default_factory=Action)
    name: str = "constant"

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        return self.action


@dataclass
class PathFollowPolicy:
    """Baseline that reads privileged observation extras (FakeEnv with privileged_obs=True).

    Steering: pure pursuit. Speed: IDM against the lead vehicle and against a red light treated
    as a stationary obstacle. Safety layer on top.
    """

    speed_limit_mps: float = 13.9
    wheelbase_m: float = 3.0
    max_steer_rad: float = 0.5
    acc: ACCController = field(default_factory=ACCController)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    name: str = "pathfollow"

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        points = obs.extras.get("lookahead_points")
        if not points:
            raise ValueError("PathFollowPolicy needs privileged observations (lookahead_points)")
        lookahead = min(20.0, max(5.0, 0.8 * obs.speed_mps + 4.0))
        steer = PurePursuit(self.wheelbase_m, lookahead, self.max_steer_rad).steer(points)
        accel = self.acc.accel(obs, self.speed_limit_mps)
        dist = obs.extras.get("next_light_distance_m")
        if obs.extras.get("next_light_red") and dist is not None and dist < LIGHT_REACT_DISTANCE_M:
            gap = max(dist - LIGHT_STOP_MARGIN_M, 0.1)
            accel = min(accel, self.acc.idm.accel(obs.speed_mps, gap, obs.speed_mps))
        throttle, brake = self.acc.pedals(accel)
        return apply_safety(Action(steer, throttle, brake), obs, self.safety)
