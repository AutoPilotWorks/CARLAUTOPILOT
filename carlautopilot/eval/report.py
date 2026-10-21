from __future__ import annotations


def make_report(agg: dict, per_episode: list[dict], title: str = "Evaluation report") -> str:
    lines = [f"# {title}", "", "## Summary", ""]
    lines.append(f"- episodes: {agg['episodes']}")
    lines.append(f"- mean driving score: {agg['mean_driving_score']:.3f}")
    lines.append(f"- min driving score: {agg['min_driving_score']:.3f}")
    lines.append(f"- success rate: {agg['success_rate']:.3f}")
    lines.append(f"- mean route completion: {agg['mean_route_completion']:.3f}")
    events = ", ".join(f"{k}={v}" for k, v in sorted(agg["events"].items())) or "none"
    lines.append(f"- events: {events}")
    lines += ["", "## Episodes", "", "| scenario | policy | termination | completion | score | success | events |", "|---|---|---|---|---|---|---|"]
    for m in per_episode:
        ev = ", ".join(f"{k}={v}" for k, v in sorted(m["events"].items())) or "-"
        lines.append(
            f"| {m['scenario']} | {m['policy']} | {m['termination']} | {m['route_completion']:.3f} | "
            f"{m['driving_score']:.3f} | {m['success']} | {ev} |"
        )
    return "\n".join(lines) + "\n"
