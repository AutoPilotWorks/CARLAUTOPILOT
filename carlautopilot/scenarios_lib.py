from __future__ import annotations

import math

from carlautopilot.core import Command
from carlautopilot.scenario import LeadVehicleSpec, RouteCommand, ScenarioSpec, TrafficLightSpec, Waypoint

SPACING_M = 2.0


def _line(x0: float, y0: float, heading: float, length: float) -> list[Waypoint]:
    n = max(1, int(length / SPACING_M))
    return [Waypoint(x0 + math.cos(heading) * length * i / n, y0 + math.sin(heading) * length * i / n) for i in range(n + 1)]


def straight(length_m: float = 300.0, speed_limit_mps: float = 13.9, name: str = "straight") -> ScenarioSpec:
    return ScenarioSpec(name=name, route=tuple(_line(0.0, 0.0, 0.0, length_m)), commands=(RouteCommand(0.0, Command.STRAIGHT),), speed_limit_mps=speed_limit_mps, max_time_s=120.0, tags=("basic",))


def curve(direction: str = "left", radius_m: float = 40.0, angle_deg: float = 90.0, lead_in_m: float = 60.0, lead_out_m: float = 60.0, speed_limit_mps: float = 6.0, name: str | None = None) -> ScenarioSpec:
    if direction not in ("left", "right"):
        raise ValueError("direction must be left or right")
    sign = 1.0 if direction == "left" else -1.0
    pts = _line(0.0, 0.0, 0.0, lead_in_m)
    cx, cy = lead_in_m, sign * radius_m
    steps = max(4, int(math.radians(angle_deg) * radius_m / SPACING_M))
    for i in range(1, steps + 1):
        th = math.radians(angle_deg) * i / steps
        pts.append(Waypoint(cx + radius_m * math.sin(th), cy - sign * radius_m * math.cos(th)))
    end = pts[-1]
    heading = sign * math.radians(angle_deg)
    pts += _line(end.x, end.y, heading, lead_out_m)[1:]
    cmd = Command.LEFT if direction == "left" else Command.RIGHT
    return ScenarioSpec(
        name=name or f"curve_{direction}",
        route=tuple(pts),
        commands=(RouteCommand(0.0, Command.STRAIGHT), RouteCommand(lead_in_m - 10.0, cmd)),
        speed_limit_mps=speed_limit_mps,
        max_time_s=120.0,
        tags=("basic", "curve"),
    )


def lead_brake(length_m: float = 400.0, name: str = "lead_brake") -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        route=tuple(_line(0.0, 0.0, 0.0, length_m)),
        lead=LeadVehicleSpec(40.0, ((0.0, 10.0), (15.0, 2.0), (30.0, 10.0))),
        max_time_s=120.0,
        meta={"initial_speed_mps": 8.0},
        tags=("acc",),
    )


def red_light(length_m: float = 300.0, light_s_m: float = 120.0, name: str = "red_light") -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        route=tuple(_line(0.0, 0.0, 0.0, length_m)),
        lights=(TrafficLightSpec(light_s_m, red_s=25.0, green_s=60.0),),
        max_time_s=150.0,
        meta={"initial_speed_mps": 8.0},
        tags=("light",),
    )


def standard_suite() -> list[ScenarioSpec]:
    return [straight(), curve("left"), curve("right"), lead_brake(), red_light()]
