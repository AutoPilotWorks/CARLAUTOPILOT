from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from carlautopilot.core import Action, Env, Observation, Policy
from carlautopilot.data.episode_store import PERCEPTION_KEYS, write_episode
from carlautopilot.models.route_steering import PERCEPT_RANGE_M
from carlautopilot.scenario import ScenarioSpec

NOISE_AR = 0.9


def perception_labels(obs: Observation) -> tuple[float, float, float, float] | None:
    """(must stop at a line in range, distance to it, lead gap, lead relative speed) from privileged observations.

    The stored key is still "light_red": a red light for my lane, an unserved stop sign, or a junction where the
    expert yields (stop_needed); "light_dist" is the nearest such stop line.
    """
    if "lookahead_points" not in obs.extras:
        return None
    rng = PERCEPT_RANGE_M
    lines = [obs.extras.get(k) for k in ("next_stop_sign_distance_m", "yield_distance_m")]
    if obs.extras.get("next_light_red"):
        lines.append(obs.extras.get("next_light_distance_m"))
    lines = [d for d in lines if d is not None and d <= rng]
    red = bool(lines)
    d = min(lines) if lines else None
    lead = obs.lead if obs.lead is not None and obs.lead.distance_m < rng else None
    return (float(red), d if red else rng, lead.distance_m if lead else rng, lead.rel_speed_mps if lead else 0.0)


def collect_episode(
    env: Env,
    expert: Policy,
    scenario: ScenarioSpec,
    root: str | Path,
    episode_id: str,
    steer_noise_std: float = 0.0,
    noise_seed: int = 0,
    max_steps: int | None = None,
    stride: int = 1,
    extra_samples: Callable[[Observation, Action], list[tuple[np.ndarray, Action]]] | None = None,
    png_compress: int = 6,
    stream_images: Callable[[Observation], list[list[np.ndarray]]] | None = None,
    stream_roots: Sequence[str | Path] = (),
) -> dict:
    """Drive with the expert and store (image, speed, command, expert action) per step.

    extra_samples(obs, action) may add (image, label) pairs for the same step, e.g. side cameras.
    stream_images(obs) returns, for each extra camera rig, the images matching this step's rows (centre then the
    extra samples); each rig is written with the same labels to the matching root in stream_roots.

    The executed steering gets temporally correlated noise, but the stored label is the clean
    expert action, so the dataset covers off-centre states and shows how to recover from them.
    """
    t_start = time.perf_counter()
    obs = env.reset(scenario)
    if obs.image is None:
        raise ValueError("the environment must provide images (render_images=True)")
    expert.reset()
    rng = np.random.default_rng(noise_seed)
    limit = max_steps if max_steps is not None else int(scenario.max_time_s / env.dt) + 10
    noise = 0.0
    rows, images, target_speeds, percepts = [], [], [], []
    streams: list[list[np.ndarray]] = []
    termination = None
    for step in range(limit):
        action = expert.act(obs).clipped()
        if step % stride == 0:
            rows.append((obs.t, obs.speed_mps, int(obs.command), action.steer, action.throttle, action.brake))
            images.append(obs.image)
            target_speeds.append(getattr(expert, "last_target_speed", None))
            percepts.append(perception_labels(obs))
            n_before = len(images) - 1
            for img, lab in extra_samples(obs, action) if extra_samples else ():
                rows.append((obs.t, obs.speed_mps, int(obs.command), lab.steer, lab.throttle, lab.brake))
                images.append(img)
                target_speeds.append(target_speeds[-1])
                percepts.append(percepts[-1])
            if stream_images is not None:
                per_rig = stream_images(obs)
                if not streams:
                    streams.extend([] for _ in per_rig)
                for k, imgs in enumerate(per_rig):
                    if len(imgs) != len(images) - n_before:
                        raise ValueError("stream images do not match the rows of this step")
                    streams[k].extend(imgs)
        noise = NOISE_AR * noise + math.sqrt(1.0 - NOISE_AR**2) * float(rng.normal(0.0, steer_noise_std)) if steer_noise_std > 0 else 0.0
        res = env.step(Action(action.steer + noise, action.throttle, action.brake).clipped())
        obs = res.obs
        if res.done:
            termination = res.termination
            break
    if not rows:
        raise ValueError("no steps recorded")
    t, speed, command, steer, throttle, brake = (np.array(c) for c in zip(*rows))
    arrays = {
        "t": t.astype(np.float64),
        "speed": speed.astype(np.float32),
        "command": command.astype(np.int64),
        "steer": steer.astype(np.float32),
        "throttle": throttle.astype(np.float32),
        "brake": brake.astype(np.float32),
    }
    if all(v is not None for v in target_speeds):
        arrays["target_speed"] = np.array(target_speeds, dtype=np.float32)
    if all(v is not None for v in percepts):
        for k, col in zip(PERCEPTION_KEYS, zip(*percepts)):
            arrays[k] = np.array(col, dtype=np.float32)
    stack = np.stack(images)
    meta = {
        "scenario_name": scenario.name,
        "seed": scenario.seed,
        "dt": env.dt,
        "image_height": int(stack.shape[1]),
        "image_width": int(stack.shape[2]),
    }
    t_sim = time.perf_counter()
    write_episode(root, episode_id, meta, arrays, stack, png_compress)
    for k, extra_root in enumerate(stream_roots):
        write_episode(extra_root, episode_id, meta, arrays, np.stack(streams[k]), png_compress)
    t_end = time.perf_counter()
    return {"episode_id": episode_id, "steps": len(rows), "termination": termination, "sim_s": round(t_sim - t_start, 1), "write_s": round(t_end - t_sim, 1)}


def collect_dataset(
    env_factory: Callable[[], Env],
    expert_factory: Callable[[], Policy],
    scenarios: Sequence[ScenarioSpec],
    root: str | Path,
    steer_noise_std: float = 0.0,
    seed: int = 0,
    prefix: str = "ep",
    stride: int = 1,
) -> list[dict]:
    infos = []
    for i, sc in enumerate(scenarios):
        env = env_factory()
        try:
            infos.append(collect_episode(env, expert_factory(), sc, root, f"{prefix}{i:04d}", steer_noise_std, seed + i, stride=stride))
        finally:
            env.close()
    return infos
