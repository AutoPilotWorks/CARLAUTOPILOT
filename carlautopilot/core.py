from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from carlautopilot.scenario import ScenarioSpec


class Command(IntEnum):
    FOLLOW = 0
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3


class EventKind:
    COLLISION_VEHICLE = "collision_vehicle"
    COLLISION_PEDESTRIAN = "collision_pedestrian"
    COLLISION_STATIC = "collision_static"
    RED_LIGHT = "red_light"
    STOP_SIGN = "stop_sign"
    OFF_ROAD = "off_road"
    LANE_DEPARTURE = "lane_departure"
    ROUTE_DEVIATION = "route_deviation"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class LeadInfo:
    distance_m: float
    rel_speed_mps: float


@dataclass(frozen=True)
class Observation:
    t: float
    speed_mps: float
    command: Command
    image: np.ndarray | None = None
    lead: LeadInfo | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def clipped(self) -> "Action":
        return Action(
            steer=float(min(1.0, max(-1.0, self.steer))),
            throttle=float(min(1.0, max(0.0, self.throttle))),
            brake=float(min(1.0, max(0.0, self.brake))),
        )


@dataclass(frozen=True)
class Event:
    t: float
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class GroundTruth:
    x: float
    y: float
    yaw_rad: float
    speed_mps: float
    route_progress_m: float
    route_length_m: float
    lateral_error_m: float


@dataclass(frozen=True)
class StepResult:
    obs: Observation
    truth: GroundTruth
    done: bool
    events: tuple[Event, ...] = ()
    termination: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


TERMINATIONS = ("route_complete", "timeout", "collision", "off_route", "blocked")


@dataclass(frozen=True)
class StepRecord:
    step: int
    t: float
    speed_mps: float
    command: int
    action: Action
    truth: GroundTruth
    events: tuple[Event, ...] = ()
    lead_distance_m: float | None = None


@dataclass(frozen=True)
class EpisodeHeader:
    scenario_name: str
    seed: int | None
    dt: float
    policy_name: str
    schema_version: int = 1


@runtime_checkable
class Env(Protocol):
    dt: float

    def reset(self, scenario: "ScenarioSpec", seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, obs: Observation) -> Action: ...
