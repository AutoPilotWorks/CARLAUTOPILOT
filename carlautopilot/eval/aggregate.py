from __future__ import annotations

from collections import Counter


def aggregate(metrics: list[dict]) -> dict:
    if not metrics:
        raise ValueError("nothing to aggregate")
    n = len(metrics)
    events: Counter[str] = Counter()
    for m in metrics:
        events.update(m["events"])
    scores = [m["driving_score"] for m in metrics]
    return {
        "episodes": n,
        "mean_driving_score": sum(scores) / n,
        "min_driving_score": min(scores),
        "success_rate": sum(1 for m in metrics if m["success"]) / n,
        "mean_route_completion": sum(m["route_completion"] for m in metrics) / n,
        "events": dict(events),
    }
