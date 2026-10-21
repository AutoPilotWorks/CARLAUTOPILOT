from __future__ import annotations

from collections import Counter

from carlautopilot.core import EpisodeHeader, EventKind, StepRecord

DEFAULT_PENALTIES = {
    EventKind.COLLISION_VEHICLE: 0.6,
    EventKind.COLLISION_PEDESTRIAN: 0.5,
    EventKind.COLLISION_STATIC: 0.65,
    EventKind.RED_LIGHT: 0.7,
    EventKind.STOP_SIGN: 0.8,
    EventKind.OFF_ROAD: 0.8,
    EventKind.LANE_DEPARTURE: 0.95,
    EventKind.ROUTE_DEVIATION: 0.8,
    EventKind.BLOCKED: 0.9,
}


def compute_metrics(
    header: EpisodeHeader,
    records: list[StepRecord],
    termination: str | None,
    penalties: dict[str, float] | None = None,
) -> dict:
    """Per-episode metrics.

    route_completion is max progress / route length, but exactly 1.0 when the environment ended the
    episode as route_complete (the environment accepts the end of the route within a small margin).

    max_decel_mps2: largest speed drop per second between consecutive records (dt = header.dt).
    mean_abs_jerk: mean |d(accel)/dt| where accel is the speed difference over dt.
    """
    if not records:
        raise ValueError("no records")
    pen = dict(DEFAULT_PENALTIES)
    if penalties:
        pen.update(penalties)
    length = records[0].truth.route_length_m
    progress = max(r.truth.route_progress_m for r in records)
    completion = min(1.0, max(0.0, progress / length)) if length > 0 else 0.0
    if termination == "route_complete":
        completion = 1.0
    counts: Counter[str] = Counter(e.kind for r in records for e in r.events)
    factor = 1.0
    for kind, n in counts.items():
        factor *= pen.get(kind, 1.0) ** n
    speeds = [r.speed_mps for r in records]
    dt = header.dt
    accels = [(b - a) / dt for a, b in zip(speeds, speeds[1:])]
    jerks = [abs(b - a) / dt for a, b in zip(accels, accels[1:])]
    leads = [r.lead_distance_m for r in records if r.lead_distance_m is not None]
    collided = any(k.startswith("collision") for k in counts)
    return {
        "scenario": header.scenario_name,
        "policy": header.policy_name,
        "seed": header.seed,
        "termination": termination,
        "route_completion": completion,
        "driving_score": completion * factor,
        "success": termination == "route_complete" and not collided,
        "events": dict(counts),
        "mean_speed_mps": sum(speeds) / len(speeds),
        "max_decel_mps2": max((-a for a in accels if a < 0), default=0.0),
        "mean_abs_jerk": sum(jerks) / len(jerks) if jerks else 0.0,
        "min_lead_distance_m": min(leads) if leads else None,
        "duration_s": records[-1].t - records[0].t,
    }
