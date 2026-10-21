from __future__ import annotations

import math
from collections.abc import Sequence


class PurePursuit:
    """Points are in the vehicle frame: x forward, y LEFT positive.

    Project convention: steer > 0 turns right, so a target on the left gives a negative steer.
    """

    def __init__(self, wheelbase_m: float = 3.0, lookahead_m: float = 8.0, max_steer_rad: float = 0.5):
        self.wheelbase_m = wheelbase_m
        self.lookahead_m = lookahead_m
        self.max_steer_rad = max_steer_rad

    def target(self, points: Sequence[tuple[float, float]]) -> tuple[float, float]:
        for x, y in points:
            if x > 0 and math.hypot(x, y) >= self.lookahead_m:
                return x, y
        if not points:
            raise ValueError("no lookahead points")
        return points[-1]

    def steer(self, points: Sequence[tuple[float, float]]) -> float:
        x, y = self.target(points)
        ld = max(math.hypot(x, y), 1e-6)
        curvature = 2.0 * y / (ld * ld)
        delta_left = math.atan(self.wheelbase_m * curvature)
        return float(min(1.0, max(-1.0, -delta_left / self.max_steer_rad)))
