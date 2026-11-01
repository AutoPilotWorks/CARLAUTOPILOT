"""CARLA-independent helpers: coordinate conversion, route commands, windowed route projection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from carlautopilot.core import Command
from carlautopilot.scenario import RouteCommand

# agents.navigation.local_planner.RoadOption values
ROAD_OPTION_TO_COMMAND = {1: Command.LEFT, 2: Command.RIGHT, 3: Command.STRAIGHT}
LANE_CHANGE_OPTIONS = {5, 6}  # CHANGELANELEFT/RIGHT: no route command for these yet
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


JUNCTION_TURN_DEG = 45.0
JUNCTION_REACH_M = 6.0


def junction_turn_options(points: Sequence[tuple[float, float]], options: Sequence[int], in_junction: Sequence[bool],
                          turn_deg: float = JUNCTION_TURN_DEG, reach_m: float = JUNCTION_REACH_M) -> list[int]:
    """GlobalRoutePlanner sometimes labels a real junction turn LANEFOLLOW (or STRAIGHT / the wrong side): 16 of 896
    junctions in the 2026-10-29 count, 3 of them off-road endings in Town03. Where the route's heading from reach_m
    before a junction run to reach_m after it turns by >= turn_deg, the run becomes LEFT/RIGHT by that turn; smaller
    turns keep the planner's option, and a run touching either route end has no heading to measure and is kept too."""
    pts = np.asarray(points, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    out = [int(o) for o in options]
    n, i = len(out), 0
    while i < n:
        if not in_junction[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and in_junction[j + 1]:
            j += 1
        # up to reach_m outside the run (fewer when the route starts/ends closer: 200_005 starts 6 m before its turn)
        a, b = max(np.searchsorted(s, s[i] - reach_m, "right") - 1, 0), min(np.searchsorted(s, s[j] + reach_m), n - 1)
        if i > 0 and j < n - 1:
            h_in = math.atan2(*(pts[i] - pts[a])[::-1])
            h_out = math.atan2(*(pts[b] - pts[j])[::-1])
            turn = math.degrees((h_out - h_in + math.pi) % (2 * math.pi) - math.pi)  # + is left (y points left)
            want = 1 if turn > 0 else 2
            if abs(turn) >= turn_deg and want not in out[i : j + 1]:  # planner already right: keep its exact extent
                out[i : j + 1] = [want] * (j - i + 1)
        i = j + 1
    return out


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


def dedupe_points(points: Sequence[tuple[float, float]], min_step_m: float = 0.05) -> list[int]:
    """Indices of points kept so that consecutive kept points are at least min_step_m apart."""
    keep = [0]
    for i in range(1, len(points)):
        a, b = points[keep[-1]], points[i]
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_step_m:
            keep.append(i)
    return keep
