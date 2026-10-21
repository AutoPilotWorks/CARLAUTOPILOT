from __future__ import annotations

import math
from dataclasses import dataclass, replace

from carlautopilot.core import Observation


@dataclass(frozen=True)
class IDM:
    """Intelligent Driver Model. dv is the closing speed: ego speed minus lead speed."""

    a_max: float = 1.5
    b: float = 2.0
    v0: float = 13.9
    T: float = 1.5
    s0: float = 2.0
    delta: float = 4.0

    def accel(self, v: float, gap: float | None = None, dv: float = 0.0) -> float:
        v = max(v, 0.0)
        free = 1.0 - (v / self.v0) ** self.delta
        if gap is None:
            return self.a_max * free
        s_star = self.s0 + max(0.0, v * self.T + v * dv / (2.0 * math.sqrt(self.a_max * self.b)))
        return self.a_max * (free - (s_star / max(gap, 1e-3)) ** 2)

    def equilibrium_gap(self, v: float) -> float:
        free = 1.0 - (max(v, 0.0) / self.v0) ** self.delta
        if free <= 0:
            return math.inf
        return (self.s0 + v * self.T) / math.sqrt(free)


@dataclass(frozen=True)
class ACCController:
    idm: IDM = IDM()
    max_brake_decel: float = 8.0

    def accel(self, obs: Observation, speed_limit: float) -> float:
        idm = replace(self.idm, v0=min(self.idm.v0, speed_limit))
        if obs.lead is None:
            return idm.accel(obs.speed_mps)
        return idm.accel(obs.speed_mps, obs.lead.distance_m, -obs.lead.rel_speed_mps)

    def act(self, obs: Observation, speed_limit: float) -> tuple[float, float]:
        a = self.accel(obs, speed_limit)
        if a >= 0:
            return min(1.0, a / self.idm.a_max), 0.0
        return 0.0, min(1.0, -a / self.max_brake_decel)
