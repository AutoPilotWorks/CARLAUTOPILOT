from __future__ import annotations

import json
from collections import deque
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
    frames_dir: str | Path | None = None,
) -> EpisodeResult:
    """frames_dir: on every event and on a failed ending, save the last FRAME_BUFFER camera frames (every FRAME_EVERY steps)
    as PNGs under frames_dir/<t>_<kind>/, so the scene can be looked at."""
    obs = env.reset(scenario, seed)
    frames: deque = deque(maxlen=FRAME_BUFFER)
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
                percept=getattr(policy, "last_percept", None),
            )
        )
        obs = res.obs
        if frames_dir is not None:
            if step % FRAME_EVERY == 0 and obs.image is not None:
                frames.append((step, obs.image, obs.extras.get("light_image")))
            kinds = [str(e.kind) for e in res.events]
            if res.done and res.termination != "route_complete":
                kinds.append(str(res.termination))
            for kind in kinds:
                _save_frames(Path(frames_dir) / f"{res.obs.t:07.2f}_{kind}", frames)
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


FRAME_EVERY, FRAME_BUFFER = 4, 20  # 5 Hz at dt 0.05, last 4 s


def _save_frames(d: Path, frames) -> None:
    from PIL import Image

    d.mkdir(parents=True, exist_ok=True)
    for step, img, light in frames:
        Image.fromarray(img).save(d / f"{step:05d}_main.png")
        if light is not None:
            Image.fromarray(light).save(d / f"{step:05d}_light.png")


def run_suite(
    env_factory: Callable[[], Env],
    policy_factory: Callable[[], Policy],
    scenarios: Sequence[ScenarioSpec],
    out_dir: str | Path,
    penalties: dict[str, float] | None = None,
    title: str = "Evaluation report",
    event_frames: bool = False,
) -> dict:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    per_episode = []
    for sc in scenarios:
        env, policy = env_factory(), policy_factory()
        try:
            result = run_episode(env, policy, sc, log_path=out / "logs" / f"{sc.name}.jsonl", penalties=penalties,
                                 frames_dir=out / "frames" / sc.name if event_frames else None)
        finally:
            env.close()
        per_episode.append(result.metrics)
    agg = aggregate(per_episode)
    (out / "metrics.json").write_text(json.dumps({"aggregate": agg, "episodes": per_episode}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(make_report(agg, per_episode, title), encoding="utf-8")
    return {"aggregate": agg, "episodes": per_episode}
