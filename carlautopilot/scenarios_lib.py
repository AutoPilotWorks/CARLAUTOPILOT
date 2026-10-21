from __future__ import annotations

import math

from carlautopilot.core import Command
from carlautopilot.scenario import LeadVehicleSpec, PedestrianSpec, RouteCommand, ScenarioSpec, StopSignSpec, TrafficLightSpec, Waypoint

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


def pedestrian_crossing(length_m: float = 300.0, ped_s_m: float = 150.0, start_t_s: float = 9.5, name: str = "pedestrian_crossing") -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        route=tuple(_line(0.0, 0.0, 0.0, length_m)),
        pedestrians=(PedestrianSpec(ped_s_m, lateral_start_m=3.5, speed_mps=1.4, start_t_s=start_t_s),),
        max_time_s=120.0,
        meta={"initial_speed_mps": 8.0},
        tags=("pedestrian",),
    )


def stop_sign(length_m: float = 300.0, sign_s_m: float = 120.0, name: str = "stop_sign") -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        route=tuple(_line(0.0, 0.0, 0.0, length_m)),
        stop_signs=(StopSignSpec(sign_s_m),),
        max_time_s=120.0,
        meta={"initial_speed_mps": 8.0},
        tags=("stop_sign",),
    )


def compose(segments: list[tuple], name: str = "composed", speed_limit_mps: float = 13.9, **kw) -> ScenarioSpec:
    """Chain ("straight", length_m) and ("left"|"right", radius_m, angle_deg) segments into one route with turn commands."""
    x = y = heading = s = 0.0
    pts = [Waypoint(0.0, 0.0)]
    cmds = [RouteCommand(0.0, Command.STRAIGHT)]
    for seg in segments:
        if seg[0] == "straight":
            pts += _line(x, y, heading, seg[1])[1:]
            x, y, s = pts[-1].x, pts[-1].y, s + seg[1]
            continue
        direction, radius, angle = seg
        if direction not in ("left", "right"):
            raise ValueError(f"unknown segment {direction}")
        sign = 1.0 if direction == "left" else -1.0
        cmds.append(RouteCommand(max(s - 10.0, cmds[-1].s_m), Command.LEFT if sign > 0 else Command.RIGHT))
        nx, ny = -math.sin(heading), math.cos(heading)
        cx, cy = x + sign * radius * nx, y + sign * radius * ny
        vx, vy = x - cx, y - cy
        steps = max(4, int(math.radians(angle) * radius / SPACING_M))
        for i in range(1, steps + 1):
            th = sign * math.radians(angle) * i / steps
            pts.append(Waypoint(cx + vx * math.cos(th) - vy * math.sin(th), cy + vx * math.sin(th) + vy * math.cos(th)))
        x, y = pts[-1].x, pts[-1].y
        heading += sign * math.radians(angle)
        s += math.radians(angle) * radius
        cmds.append(RouteCommand(s, Command.STRAIGHT))
    return ScenarioSpec(name=name, route=tuple(pts), commands=tuple(cmds), speed_limit_mps=speed_limit_mps, **kw)


def delivery_route(name: str = "delivery_route") -> ScenarioSpec:
    """Simulated mini delivery: three turns, a red light, a stop sign and a slow lead vehicle."""
    return compose(
        [("straight", 150.0), ("right", 40.0, 90.0), ("straight", 120.0), ("left", 40.0, 90.0), ("straight", 100.0), ("left", 40.0, 90.0), ("straight", 150.0)],
        name=name,
        lights=(TrafficLightSpec(100.0, red_s=20.0, green_s=60.0),),
        stop_signs=(StopSignSpec(300.0),),
        lead=LeadVehicleSpec(60.0, ((0.0, 9.0),)),
        max_time_s=200.0,
        meta={"initial_speed_mps": 6.0},
        tags=("route", "full"),
    )


def standard_suite() -> list[ScenarioSpec]:
    return [straight(), curve("left"), curve("right"), lead_brake(), red_light(), pedestrian_crossing(), stop_sign(), delivery_route()]


def random_scenarios(n: int, seed: int = 0, families: tuple[str, ...] = ("straight", "curve", "lead_brake")) -> list[ScenarioSpec]:
    import random

    rng = random.Random(seed)
    out: list[ScenarioSpec] = []
    for i in range(n):
        family = families[i % len(families)]
        name = f"{family}_{i:03d}"
        if family == "straight":
            sc = straight(rng.uniform(120.0, 260.0), speed_limit_mps=rng.uniform(8.0, 13.9), name=name)
        elif family == "curve":
            sc = curve(
                rng.choice(["left", "right"]),
                radius_m=rng.uniform(25.0, 60.0),
                angle_deg=rng.uniform(45.0, 120.0),
                lead_in_m=rng.uniform(40.0, 80.0),
                lead_out_m=rng.uniform(40.0, 80.0),
                speed_limit_mps=rng.uniform(5.0, 7.0),
                name=name,
            )
        elif family == "lead_brake":
            base = lead_brake(rng.uniform(250.0, 400.0), name=name)
            gap = rng.uniform(25.0, 50.0)
            slow = rng.uniform(1.0, 4.0)
            out.append(
                ScenarioSpec(
                    name=base.name,
                    route=base.route,
                    lead=LeadVehicleSpec(gap, ((0.0, 10.0), (rng.uniform(10.0, 20.0), slow), (rng.uniform(25.0, 40.0), 10.0))),
                    max_time_s=base.max_time_s,
                    meta=base.meta,
                    tags=base.tags,
                    seed=i,
                )
            )
            continue
        elif family == "red_light":
            sc = red_light(rng.uniform(250.0, 350.0), rng.uniform(80.0, 150.0), name=name)
        elif family == "stop_sign":
            sc = stop_sign(rng.uniform(250.0, 350.0), rng.uniform(80.0, 150.0), name=name)
        elif family == "route":
            segs: list[tuple] = [("straight", rng.uniform(80.0, 150.0))]
            for _ in range(rng.randint(2, 4)):
                segs.append((rng.choice(["left", "right"]), rng.uniform(30.0, 50.0), rng.choice([60.0, 90.0])))
                segs.append(("straight", rng.uniform(60.0, 120.0)))
            sc = compose(segs, name=name, max_time_s=250.0, meta={"initial_speed_mps": 6.0}, tags=("route",))
        else:
            raise ValueError(f"unknown family {family}")
        out.append(ScenarioSpec(**{**sc.__dict__, "seed": i}))
    return out
