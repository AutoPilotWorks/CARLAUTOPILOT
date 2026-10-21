from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from carlautopilot.core import Env, EpisodeHeader, Policy, StepRecord
from carlautopilot.eval import EpisodeLogWriter, aggregate, compute_metrics, make_report
from carlautopilot.scenario import ScenarioSpec


@dataclass
class EpisodeResult:
    header: EpisodeHeader
    records: list[StepRecord]
    termination: str | None
    metrics: dict


def run_episode(
    env: Env,
    policy: Policy,
    scenario: ScenarioSpec,
    seed: int | None = None,
    log_path: str | Path | None = None,
    max_steps: int | None = None,
    penalties: dict[str, float] | None = None,
) -> EpisodeResult:
    obs = env.reset(scenario, seed)
    policy.reset()
    header = EpisodeHeader(scenario_name=scenario.name, seed=scenario.seed if seed is None else seed, dt=env.dt, policy_name=policy.name)
    limit = max_steps if max_steps is not None else int(scenario.max_time_s / env.dt) + 10
    records: list[StepRecord] = []
    termination: str | None = None
    for step in range(limit):
        action = policy.act(obs).clipped()
        res = env.step(action)
        records.append(
            StepRecord(
                step=step,
                t=res.obs.t,
                speed_mps=res.truth.speed_mps,
                command=int(res.obs.command),
                action=action,
                truth=res.truth,
                events=res.events,
                lead_distance_m=None if res.obs.lead is None else res.obs.lead.distance_m,
            )
        )
        obs = res.obs
        if res.done:
            termination = res.termination
            break
    metrics = compute_metrics(header, records, termination, penalties)
    if log_path is not None:
        with EpisodeLogWriter(log_path) as w:
            w.write_header(header)
            for r in records:
                w.write(r)
    return EpisodeResult(header, records, termination, metrics)


def run_suite(
    env_factory: Callable[[], Env],
    policy_factory: Callable[[], Policy],
    scenarios: Sequence[ScenarioSpec],
    out_dir: str | Path,
    penalties: dict[str, float] | None = None,
    title: str = "Evaluation report",
) -> dict:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    per_episode = []
    for sc in scenarios:
        env, policy = env_factory(), policy_factory()
        try:
            result = run_episode(env, policy, sc, log_path=out / "logs" / f"{sc.name}.jsonl", penalties=penalties)
        finally:
            env.close()
        per_episode.append(result.metrics)
    agg = aggregate(per_episode)
    (out / "metrics.json").write_text(json.dumps({"aggregate": agg, "episodes": per_episode}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(make_report(agg, per_episode, title), encoding="utf-8")
    return {"aggregate": agg, "episodes": per_episode}
