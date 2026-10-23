from __future__ import annotations

import math
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
    max_lat_accel: float | None = None
    lookahead_gain: float = 0.8
    lookahead_offset_m: float = 4.0
    lookahead_min_m: float = 5.0
    name: str = "pathfollow"

    def reset(self) -> None:
        pass

    def speed_cap(self, points) -> float:
        """Speed limit from the sharpest curvature among the lookahead points (pure-pursuit arc through each)."""
        if self.max_lat_accel is None:
            return self.speed_limit_mps
        kappa = max((abs(2.0 * y / (x * x + y * y)) for x, y in points if x > 0), default=0.0)
        return min(self.speed_limit_mps, math.sqrt(self.max_lat_accel / max(kappa, 1e-6)))

    def act(self, obs: Observation) -> Action:
        points = obs.extras.get("lookahead_points")
        if not points:
            raise ValueError("PathFollowPolicy needs privileged observations (lookahead_points)")
        lookahead = min(20.0, max(self.lookahead_min_m, self.lookahead_gain * obs.speed_mps + self.lookahead_offset_m))
        steer = PurePursuit(self.wheelbase_m, lookahead, self.max_steer_rad).steer(points)
        # road-geometry speed (curvature cap), stored as a label so a model can learn speed later
        self.last_target_speed = self.speed_cap(points)
        accel = self.acc.accel(obs, self.last_target_speed)
        stops = []
        dist = obs.extras.get("next_light_distance_m")
        if obs.extras.get("next_light_red") and dist is not None:
            stops.append(dist)
        if "next_stop_sign_distance_m" in obs.extras:
            stops.append(obs.extras["next_stop_sign_distance_m"])
        for dist in stops:
            if dist < LIGHT_REACT_DISTANCE_M:
                gap = max(dist - LIGHT_STOP_MARGIN_M, 0.1)
                accel = min(accel, self.acc.idm.accel(obs.speed_mps, gap, obs.speed_mps))
        throttle, brake = self.acc.pedals(accel)
        return apply_safety(Action(steer, throttle, brake), obs, self.safety)
