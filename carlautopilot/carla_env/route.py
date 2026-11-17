"""CARLA-independent helpers: coordinate conversion, route commands, windowed route projection."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np

from carlautopilot.core import Command, LeadInfo
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

    def project_many(self, xy: np.ndarray, s_min: float = -math.inf, s_max: float = math.inf) -> tuple[np.ndarray, np.ndarray]:
        """project() for (N, 2) points at once -> (s, lateral left-positive); the radar's thousands of points per second
        made the per-point loop the slow part (p99 8-10 ms, 11-17)."""
        idx = np.nonzero((self.cum[1:] >= s_min) & (self.cum[:-1] <= s_max))[0]
        if len(idx) == 0:
            idx = np.arange(len(self.seg_len))
        a, d, length = self.points[idx], self.seg_dir[idx], self.seg_len[idx]
        rel = xy[:, None, :] - a[None]  # (N, M, 2)
        t = np.clip(np.einsum("nmj,mj->nm", rel, d), 0.0, length)
        off = rel - d[None] * t[..., None]
        dist = np.hypot(off[..., 0], off[..., 1])
        k = np.argmin(dist, axis=1)
        n = np.arange(len(xy))
        cross = d[k, 0] * rel[n, k, 1] - d[k, 1] * rel[n, k, 0]
        return self.cum[idx][k] + t[n, k], dist[n, k] * np.where(cross >= 0, 1.0, -1.0)

    def point_at(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        i = int(min(np.searchsorted(self.cum, s, side="right") - 1, len(self.seg_len) - 1))
        p = self.points[i] + self.seg_dir[i] * (s - self.cum[i])
        return float(p[0]), float(p[1])

    def heading_at(self, s: float) -> float:
        i = int(min(max(np.searchsorted(self.cum, s, side="right") - 1, 0), len(self.seg_len) - 1))
        return math.atan2(self.seg_dir[i, 1], self.seg_dir[i, 0])

    def body_ahead(self, corners: Sequence[tuple[float, float]], s0: float, s_max: float, half_m: float, step_m: float = 0.25) -> float | None:
        """Smallest route position, ahead of s0, at which a body's footprint outline lies within half_m of the route.
        The outline, not the centre: a bus turning off the route can have its centre beside the lane while its tail
        still blocks it (Town03_Opt_111_022, 2026-10-31). Not the corners alone either: a body wider than the corridor
        centred on it has every corner outside (11-02: the expert drove into 11 of 40 Town01 routes), and taking the
        nearest corner of such a body put the gap at a corner far to the side, short of where it blocks (11-17)."""
        p = np.unique(np.round(np.asarray(corners, dtype=float), 3), axis=0)  # get_world_vertices: 8 corners, 4 in plan
        c = p.mean(axis=0)
        p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
        edges = []
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            k = max(int(np.ceil(np.hypot(*(b - a)) / step_m)), 1)
            edges.append(a + (b - a) * np.linspace(0.0, 1.0, k, endpoint=False)[:, None])
        s, lat = self.project_many(np.concatenate(edges), s0, s_max)
        ahead = s[(s > s0) & (np.abs(lat) < half_m)]
        return float(ahead.min()) if len(ahead) else None


def points_seen_from(points: Sequence[tuple[float, float]], dy_m: float, dyaw_deg: float) -> list[tuple[float, float]]:
    """Vehicle-frame points (x forward, y left) as seen from a pose shifted dy_m left and yawed dyaw_deg left."""
    c, s = math.cos(-math.radians(dyaw_deg)), math.sin(-math.radians(dyaw_deg))
    return [(x * c - (y - dy_m) * s, x * s + (y - dy_m) * c) for x, y in points]


def radar_local(points: np.ndarray) -> np.ndarray:
    """CARLA radar detections (N, 4) [velocity, azimuth, altitude, depth] -> (N, 3) sensor-frame x forward, y right, z up."""
    _, az, alt, depth = points.T
    return np.stack([depth * np.cos(alt) * np.cos(az), depth * np.cos(alt) * np.sin(az), depth * np.sin(alt)], axis=1)


def radar_obstacle_mask(local: np.ndarray, mount_z: float, min_h: float = 0.3, max_h: float = 2.5) -> np.ndarray:
    """Detections between min_h and max_h above the road: drops road-surface returns and overhead signs or bridges."""
    h = local[:, 2] + mount_z
    return (h > min_h) & (h < max_h)


class RadarTrack:
    """A radar obstacle counts once seen in `confirm` of the last `window` ticks, then stays through up to `hold_s` of
    missed ticks with its gap shrinking by our own travel (a rule; drops one-tick ghosts and bridges flicker). Two in a
    row was too strict for sparse far returns: 15-20 m vehicles fell from 82% raw to 58% tracked (Town10HD, 11-17)."""

    def __init__(self, confirm: int = 2, hold_s: float = 0.3, window: int = 3):
        self.confirm, self.hold_s = confirm, hold_s
        self.recent: deque[bool] = deque(maxlen=window)
        self.age, self.last = 0.0, None

    def update(self, det: LeadInfo | None, speed_mps: float, dt: float) -> LeadInfo | None:
        self.recent.append(det is not None)
        if det is not None:
            if self.last is not None or sum(self.recent) >= self.confirm:
                self.last, self.age = det, 0.0
                return det
            return None
        if self.last is not None and self.age + dt <= self.hold_s + 1e-9:
            self.age += dt
            self.last = LeadInfo(distance_m=max(self.last.distance_m - speed_mps * dt, 0.0), rel_speed_mps=self.last.rel_speed_mps)
            return self.last
        self.last = None
        return None


def dedupe_points(points: Sequence[tuple[float, float]], min_step_m: float = 0.05) -> list[int]:
    """Indices of points kept so that consecutive kept points are at least min_step_m apart."""
    keep = [0]
    for i in range(1, len(points)):
        a, b = points[keep[-1]], points[i]
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_step_m:
            keep.append(i)
    return keep


def lead_brake_plan(route: WindowedRoute, rng, n: int, max_time_s: float = 400.0) -> list[dict]:
    """Cars ahead on the ego's route that drive slower than it (so it catches up) and brake to a stop now and then
    (Euro NCAP CCRm/CCRb-like; 11-02: moving with something within 8 m was ~0.5% of the smoke data, bar 1%). Each gets
    route s (25-45 m ahead of the start, 15 m apart), a desired speed and brake windows (start s, duration s) in
    episode time."""
    plans = []
    for k in range(n):
        brakes, t = [], rng.uniform(10.0, 20.0)
        while t < max_time_s:
            d = rng.uniform(1.5, 5.0)
            brakes.append((t, d))
            t += d + rng.uniform(10.0, 25.0)
        plans.append({"s": min(rng.uniform(25.0, 45.0) + 15.0 * k, max(route.length - 20.0, 0.0)), "speed_kmh": rng.uniform(18.0, 26.0), "brakes": brakes})
    return plans


# along walkers' distance right of the route centre: their body outline (~0.35 m) stays past the 1.5 m pedestrian
# corridor (env.WALKER_BODY_HALF_M), so the expert passes instead of trailing them (11-02 probe: at 1.4 m they counted
# as leads; 11-17: at 1.75 m inside a 1.8 m corridor the ego trailed one for 75 s)
ALONG_LAT_M = (2.0, 2.5)


def path_walker_plan(route: WindowedRoute, rng, n: int, lane_half_m: float = 1.75) -> list[dict]:
    """Scripted pedestrians in the ego's path (Euro NCAP-like, 11-01: A hit one walking along the lane edge it never
    saw in its data, where only 0.34% of frames had something within 8 m while moving). Half walk along the right lane
    edge just outside the in-path corridor (ALONG_LAT_M: the ego passes them close instead of trailing them at walking
    pace, 11-02 smoke), either way along the road; half cross the road ahead once the ego is within their trigger
    distance. Positions are route s and lateral offset (left +); the env turns them into CARLA actors."""
    plans = []
    for k in range(n):
        s = rng.uniform(40.0, max(41.0, route.length - 30.0))
        h = route.heading_at(s)
        if k % 2 == 0:
            # ponytail: a straight walk from s; on a curve the walker can drift into the lane (the ego then treats it as a lead)
            lat = -rng.uniform(*ALONG_LAT_M)
            back = rng.random() < 0.5
            plans.append({"mode": "along", "s": s, "lat": lat, "speed": rng.uniform(0.8, 1.5), "heading": h + (math.pi if back else 0.0), "trigger_m": math.inf})
        else:
            side = rng.choice((-1.0, 1.0))
            plans.append({"mode": "cross", "s": s, "lat": side * (lane_half_m + 2.5), "speed": rng.uniform(1.0, 1.8),
                          "heading": h + side * math.pi / 2.0 * -1.0, "trigger_m": rng.uniform(25.0, 40.0)})
    return plans
