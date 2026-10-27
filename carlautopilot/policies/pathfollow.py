from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from carlautopilot.control import ACCController, PurePursuit, SafetyConfig, apply_safety
from carlautopilot.core import Action, Observation

LIGHT_STOP_MARGIN_M = 3.0
LIGHT_REACT_DISTANCE_M = 80.0


def stop_accel(acc: ACCController, speed: float, dist: float) -> float:
    """IDM deceleration toward a stop line dist m ahead, treated as a stationary obstacle."""
    return acc.idm.accel(speed, max(dist - LIGHT_STOP_MARGIN_M, 0.1), speed)


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
    # collection only: after a stop reason clears (light turns green, stop sign served, crossing clear) wait a random
    # 0..start_delay_max_s like a driver's reaction time; the frames standing at a line with no stop needed are rare
    # otherwise (97% of standstill frames were labelled stop), and the model learned 'at a stop line => stop'
    start_delay_max_s: float = 0.0
    start_delay_seed: int = 0
    name: str = "pathfollow"

    def reset(self) -> None:
        self._rng = random.Random(self.start_delay_seed)
        self._waiting = False
        self._release_at: float | None = None

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
        for key in ("next_stop_sign_distance_m", "yield_distance_m"):
            if key in obs.extras:
                stops.append(obs.extras[key])
        stops = [d for d in stops if d < LIGHT_REACT_DISTANCE_M]
        for dist in stops:
            accel = min(accel, stop_accel(self.acc, obs.speed_mps, dist))
        if self.start_delay_max_s > 0:
            accel = self._delayed_start(obs, bool(stops), accel)
        throttle, brake = self.acc.pedals(accel)
        return apply_safety(Action(steer, throttle, brake), obs, self.safety)

    def _delayed_start(self, obs: Observation, stop_reason: bool, accel: float) -> float:
        if not hasattr(self, "_rng"):
            self.reset()
        standing = obs.speed_mps < 0.3
        if stop_reason:
            self._waiting, self._release_at = standing, None
        elif self._waiting and standing:
            if self._release_at is None:
                self._release_at = obs.t + self._rng.uniform(0.0, self.start_delay_max_s)
            if obs.t < self._release_at:
                return -self.acc.max_brake_decel
            self._waiting = False
        else:
            self._waiting = False
        return accel
