from __future__ import annotations

import math

import numpy as np

from carlautopilot.core import (
    Action,
    Command,
    Event,
    EventKind,
    GroundTruth,
    LeadInfo,
    Observation,
    StepResult,
)
from carlautopilot.scenario import ScenarioSpec

LOOKAHEAD_OFFSETS_M = (3.0, 6.0, 9.0, 12.0, 15.0, 20.0, 25.0, 30.0)
LEAD_VISIBLE_M = 100.0
OFF_ROAD_LATERAL_M = 2.0
OFF_ROAD_SECONDS = 2.0
BLOCKED_SPEED_MPS = 0.1
BLOCKED_SECONDS = 30.0
RED_LIGHT_SPEED_MPS = 0.5
ROUTE_END_MARGIN_M = 1.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Route:
    def __init__(self, scenario: ScenarioSpec):
        pts = np.array([[w.x, w.y] for w in scenario.route], dtype=float)
        seg = pts[1:] - pts[:-1]
        self.points = pts
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        if np.any(self.seg_len <= 0):
            raise ValueError("route has a zero-length segment")
        self.seg_dir = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        p = np.array([x, y])
        rel = p - self.points[:-1]
        t = np.clip(np.einsum("ij,ij->i", rel, self.seg_dir), 0.0, self.seg_len)
        nearest = self.points[:-1] + self.seg_dir * t[:, None]
        dist = np.hypot(*(p - nearest).T)
        i = int(np.argmin(dist))
        cross = self.seg_dir[i, 0] * rel[i, 1] - self.seg_dir[i, 1] * rel[i, 0]
        lateral = float(dist[i]) * (1.0 if cross >= 0 else -1.0)
        heading = math.atan2(self.seg_dir[i, 1], self.seg_dir[i, 0])
        return float(self.cum[i] + t[i]), lateral, heading

    def point_at(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        i = int(min(np.searchsorted(self.cum, s, side="right") - 1, len(self.seg_len) - 1))
        p = self.points[i] + self.seg_dir[i] * (s - self.cum[i])
        return float(p[0]), float(p[1])

    def heading_at(self, s: float) -> float:
        i = int(min(max(np.searchsorted(self.cum, s, side="right") - 1, 0), len(self.seg_len) - 1))
        return math.atan2(self.seg_dir[i, 1], self.seg_dir[i, 0])


class FakeEnv:
    """Kinematic bicycle on a polyline route. steer > 0 turns RIGHT (yaw decreases).

    The lead vehicle speed profile is piecewise constant: speed(t) is the value of the last
    (t_k, v_k) with t_k <= t. Lateral error is positive to the left of the route.
    """

    def __init__(
        self,
        dt: float = 0.05,
        wheelbase_m: float = 3.0,
        max_steer_rad: float = 0.5,
        max_accel: float = 3.0,
        max_brake_decel: float = 8.0,
        drag: float = 0.02,
        privileged_obs: bool = False,
        speed_noise_std: float = 0.0,
    ):
        self.dt = dt
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = max_steer_rad
        self.max_accel = max_accel
        self.max_brake_decel = max_brake_decel
        self.drag = drag
        self.privileged_obs = privileged_obs
        self.speed_noise_std = speed_noise_std
        self._scenario: ScenarioSpec | None = None

    def reset(self, scenario: ScenarioSpec, seed: int | None = None) -> Observation:
        scenario.validate()
        self._scenario = scenario
        self._route = _Route(scenario)
        self._rng = np.random.default_rng(scenario.seed if seed is None else seed)
        x, y = self._route.point_at(0.0)
        self.x, self.y = x, y
        self.yaw = self._route.heading_at(0.0)
        self.v = float(scenario.meta.get("initial_speed_mps", 0.0))
        self.t = 0.0
        self._step = 0
        self._s, self._lat, _ = self._route.project(self.x, self.y)
        self._lead_s = scenario.lead.initial_gap_m if scenario.lead else None
        self._lead_v = self._lead_speed_at(0.0)
        self._light_events_done: set[int] = set()
        self._off_road_time = 0.0
        self._blocked_time = 0.0
        self._done = False
        return self._observe()

    def close(self) -> None:
        self._scenario = None

    def _lead_speed_at(self, t: float) -> float:
        if self._scenario is None or self._scenario.lead is None:
            return 0.0
        v = self._scenario.lead.speed_profile[0][1]
        for tk, vk in self._scenario.lead.speed_profile:
            if tk <= t:
                v = vk
        return v

    def light_is_red(self, index: int, t: float) -> bool:
        light = self._scenario.lights[index]
        cycle = light.red_s + light.green_s
        return ((t + light.offset_s) % cycle) < light.red_s

    def _command(self) -> Command:
        cmd = Command.FOLLOW
        for rc in self._scenario.commands:
            if rc.s_m <= self._s:
                cmd = rc.command
        return cmd

    def _lead_info(self) -> LeadInfo | None:
        if self._lead_s is None:
            return None
        gap = self._lead_s - self._s
        if 0.0 < gap <= LEAD_VISIBLE_M:
            return LeadInfo(distance_m=gap, rel_speed_mps=self._lead_v - self.v)
        return None

    def _extras(self) -> dict:
        if not self.privileged_obs:
            return {}
        _, _, route_heading = self._route.project(self.x, self.y)
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        pts = []
        for d in LOOKAHEAD_OFFSETS_M:
            px, py = self._route.point_at(self._s + d)
            dx, dy = px - self.x, py - self.y
            pts.append((dx * c - dy * s, dx * s + dy * c))
        extras = {
            "heading_error_rad": _wrap(self.yaw - route_heading),
            "lateral_error_m": self._lat,
            "lookahead_points": pts,
        }
        ahead = [(l.s_m - self._s, i) for i, l in enumerate(self._scenario.lights) if l.s_m - self._s > 0]
        if ahead:
            dist, i = min(ahead)
            if dist <= LEAD_VISIBLE_M:
                extras["next_light_distance_m"] = dist
                extras["next_light_red"] = self.light_is_red(i, self.t)
        return extras

    def _observe(self) -> Observation:
        speed = self.v
        if self.speed_noise_std > 0:
            speed = max(0.0, speed + float(self._rng.normal(0.0, self.speed_noise_std)))
        return Observation(t=self.t, speed_mps=speed, command=self._command(), image=None, lead=self._lead_info(), extras=self._extras())

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            x=self.x,
            y=self.y,
            yaw_rad=self.yaw,
            speed_mps=self.v,
            route_progress_m=self._s,
            route_length_m=self._route.length,
            lateral_error_m=self._lat,
        )

    def step(self, action: Action) -> StepResult:
        if self._scenario is None:
            raise RuntimeError("call reset() first")
        if self._done:
            raise RuntimeError("episode finished; call reset()")
        a = action.clipped()
        dt = self.dt
        delta = -a.steer * self.max_steer_rad
        accel = a.throttle * self.max_accel - a.brake * self.max_brake_decel - self.drag * self.v
        prev_s = self._s
        prev_t = self.t
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw = _wrap(self.yaw + self.v / self.wheelbase_m * math.tan(delta) * dt)
        self.v = max(0.0, self.v + accel * dt)
        self.t = prev_t + dt
        self._step += 1
        self._s, self._lat, _ = self._route.project(self.x, self.y)
        if self._lead_s is not None:
            self._lead_s += self._lead_v * dt
            self._lead_v = self._lead_speed_at(self.t)

        events: list[Event] = []
        termination: str | None = None

        if self._lead_s is not None and self._lead_s - self._s <= 0.0:
            events.append(Event(self.t, EventKind.COLLISION_VEHICLE, "lead vehicle"))
            termination = "collision"

        for i, light in enumerate(self._scenario.lights):
            if i in self._light_events_done:
                continue
            if prev_s < light.s_m <= self._s and self.light_is_red(i, self.t) and self.v > RED_LIGHT_SPEED_MPS:
                events.append(Event(self.t, EventKind.RED_LIGHT, f"light {i}"))
                self._light_events_done.add(i)

        if abs(self._lat) > OFF_ROAD_LATERAL_M:
            self._off_road_time += dt
        else:
            self._off_road_time = 0.0
        if termination is None and self._off_road_time >= OFF_ROAD_SECONDS:
            events.append(Event(self.t, EventKind.OFF_ROAD, f"lateral {self._lat:.2f} m"))
            termination = "off_route"

        if self.v < BLOCKED_SPEED_MPS:
            self._blocked_time += dt
        else:
            self._blocked_time = 0.0
        if termination is None and self._blocked_time >= BLOCKED_SECONDS:
            events.append(Event(self.t, EventKind.BLOCKED, "stopped too long"))
            termination = "blocked"

        if termination is None and self._s >= self._route.length - ROUTE_END_MARGIN_M:
            termination = "route_complete"
        if termination is None and self.t >= self._scenario.max_time_s:
            termination = "timeout"

        self._done = termination is not None
        return StepResult(
            obs=self._observe(),
            truth=self.ground_truth(),
            done=self._done,
            events=tuple(events),
            termination=termination,
        )
