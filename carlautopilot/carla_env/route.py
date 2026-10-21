"""CARLA-independent helpers: coordinate conversion, route commands, windowed route projection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from carlautopilot.core import Command
from carlautopilot.scenario import RouteCommand

# agents.navigation.local_planner.RoadOption values
ROAD_OPTION_TO_COMMAND = {1: Command.LEFT, 2: Command.RIGHT, 3: Command.STRAIGHT}
COMMAND_LEAD_M = 10.0


def carla_to_xy(x: float, y: float) -> tuple[float, float]:
    """CARLA is left-handed (y points right); the project uses y north/left, so y flips."""
    return x, -y


def carla_yaw_to_rad(yaw_deg: float) -> float:
    return -math.radians(yaw_deg)


def commands_from_options(s: Sequence[float], options: Sequence[int], lead_m: float = COMMAND_LEAD_M) -> tuple[RouteCommand, ...]:
    """Each run of junction options (LEFT/RIGHT/STRAIGHT) becomes a command starting lead_m before it, then FOLLOW."""
    out: list[RouteCommand] = []
    i, n = 0, len(options)
    while i < n:
        cmd = ROAD_OPTION_TO_COMMAND.get(int(options[i]))
        if cmd is None:
            i += 1
            continue
        j = i
        while j + 1 < n and int(options[j + 1]) == int(options[i]):
            j += 1
        start = max(s[i] - lead_m, out[-1].s_m if out else 0.0)
        out.append(RouteCommand(start, cmd))
        out.append(RouteCommand(max(s[j], start), Command.FOLLOW))
        i = j + 1
    return tuple(out)


class WindowedRoute:
    """Polyline route whose projection is restricted to an s-window, so a route that passes near itself does not jump."""

    def __init__(self, points: Sequence[tuple[float, float]]):
        pts = np.asarray(points, dtype=float)
        seg = pts[1:] - pts[:-1]
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        if np.any(self.seg_len <= 0):
            raise ValueError("route has a zero-length segment")
        self.points = pts
        self.seg_dir = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])

    def project(self, x: float, y: float, s_min: float = -math.inf, s_max: float = math.inf) -> tuple[float, float, float]:
        """Returns (s, lateral left-positive, distance to the route) using only segments overlapping [s_min, s_max]."""
        idx = np.nonzero((self.cum[1:] >= s_min) & (self.cum[:-1] <= s_max))[0]
        if len(idx) == 0:
            idx = np.arange(len(self.seg_len))
        p = np.array([x, y])
        rel = p - self.points[idx]
        t = np.clip(np.einsum("ij,ij->i", rel, self.seg_dir[idx]), 0.0, self.seg_len[idx])
        nearest = self.points[idx] + self.seg_dir[idx] * t[:, None]
        dist = np.hypot(*(p - nearest).T)
        k = int(np.argmin(dist))
        i = idx[k]
        cross = self.seg_dir[i, 0] * rel[k, 1] - self.seg_dir[i, 1] * rel[k, 0]
        return float(self.cum[i] + t[k]), float(dist[k]) * (1.0 if cross >= 0 else -1.0), float(dist[k])

    def point_at(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        i = int(min(np.searchsorted(self.cum, s, side="right") - 1, len(self.seg_len) - 1))
        p = self.points[i] + self.seg_dir[i] * (s - self.cum[i])
        return float(p[0]), float(p[1])

    def heading_at(self, s: float) -> float:
        i = int(min(max(np.searchsorted(self.cum, s, side="right") - 1, 0), len(self.seg_len) - 1))
        return math.atan2(self.seg_dir[i, 1], self.seg_dir[i, 0])


def points_seen_from(points: Sequence[tuple[float, float]], dy_m: float, dyaw_deg: float) -> list[tuple[float, float]]:
    """Vehicle-frame points (x forward, y left) as seen from a pose shifted dy_m left and yawed dyaw_deg left."""
    c, s = math.cos(-math.radians(dyaw_deg)), math.sin(-math.radians(dyaw_deg))
    return [(x * c - (y - dy_m) * s, x * s + (y - dy_m) * c) for x, y in points]


def dedupe_points(points: Sequence[tuple[float, float]], min_step_m: float = 0.05) -> list[int]:
    """Indices of points kept so that consecutive kept points are at least min_step_m apart."""
    keep = [0]
    for i in range(1, len(points)):
        a, b = points[keep[-1]], points[i]
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_step_m:
            keep.append(i)
    return keep
