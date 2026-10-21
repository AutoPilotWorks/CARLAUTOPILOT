from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from carlautopilot.core import Command


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float


@dataclass(frozen=True)
class RouteCommand:
    s_m: float
    command: Command


@dataclass(frozen=True)
class LeadVehicleSpec:
    initial_gap_m: float
    speed_profile: tuple[tuple[float, float], ...] = ((0.0, 10.0),)


@dataclass(frozen=True)
class TrafficLightSpec:
    s_m: float
    red_s: float = 10.0
    green_s: float = 10.0
    offset_s: float = 0.0


@dataclass(frozen=True)
class PedestrianSpec:
    """Crosses the road at route position s_m, walking from lateral_start_m (left positive) to the opposite side."""

    s_m: float
    lateral_start_m: float
    speed_mps: float = 1.4
    start_t_s: float = 0.0


@dataclass(frozen=True)
class StopSignSpec:
    s_m: float


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    route: tuple[Waypoint, ...]
    commands: tuple[RouteCommand, ...] = ()
    lead: LeadVehicleSpec | None = None
    lights: tuple[TrafficLightSpec, ...] = ()
    pedestrians: tuple[PedestrianSpec, ...] = ()
    stop_signs: tuple[StopSignSpec, ...] = ()
    max_time_s: float = 120.0
    speed_limit_mps: float = 13.9
    seed: int = 0
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.route) < 2:
            raise ValueError("route needs at least 2 waypoints")
        if self.max_time_s <= 0 or self.speed_limit_mps <= 0:
            raise ValueError("max_time_s and speed_limit_mps must be positive")
        prev = -1.0
        for rc in self.commands:
            if rc.s_m < prev:
                raise ValueError("commands must be sorted by s_m")
            prev = rc.s_m
        if self.lead is not None:
            if self.lead.initial_gap_m <= 0:
                raise ValueError("lead.initial_gap_m must be positive")
            times = [t for t, _ in self.lead.speed_profile]
            if not self.lead.speed_profile or times != sorted(times):
                raise ValueError("lead.speed_profile must be non-empty and sorted by time")
        for light in self.lights:
            if light.red_s <= 0 or light.green_s <= 0:
                raise ValueError("traffic light phases must be positive")
        for ped in self.pedestrians:
            if ped.speed_mps <= 0 or ped.lateral_start_m == 0 or ped.start_t_s < 0:
                raise ValueError("pedestrian needs positive speed, nonzero lateral_start_m and start_t_s >= 0")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["commands"] = [{"s_m": rc.s_m, "command": int(rc.command)} for rc in self.commands]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ScenarioSpec":
        lead = d.get("lead")
        spec = ScenarioSpec(
            name=d["name"],
            route=tuple(Waypoint(float(w["x"]), float(w["y"])) for w in d["route"]),
            commands=tuple(RouteCommand(float(c["s_m"]), Command(int(c["command"]))) for c in d.get("commands", [])),
            lead=None
            if lead is None
            else LeadVehicleSpec(
                float(lead["initial_gap_m"]),
                tuple((float(t), float(v)) for t, v in lead.get("speed_profile", [(0.0, 10.0)])),
            ),
            lights=tuple(
                TrafficLightSpec(float(l["s_m"]), float(l.get("red_s", 10.0)), float(l.get("green_s", 10.0)), float(l.get("offset_s", 0.0)))
                for l in d.get("lights", [])
            ),
            pedestrians=tuple(
                PedestrianSpec(float(q["s_m"]), float(q["lateral_start_m"]), float(q.get("speed_mps", 1.4)), float(q.get("start_t_s", 0.0)))
                for q in d.get("pedestrians", [])
            ),
            stop_signs=tuple(StopSignSpec(float(q["s_m"])) for q in d.get("stop_signs", [])),
            max_time_s=float(d.get("max_time_s", 120.0)),
            speed_limit_mps=float(d.get("speed_limit_mps", 13.9)),
            seed=int(d.get("seed", 0)),
            tags=tuple(d.get("tags", ())),
            meta=dict(d.get("meta", {})),
        )
        spec.validate()
        return spec

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "ScenarioSpec":
        return ScenarioSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
