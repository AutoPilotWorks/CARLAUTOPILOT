from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from carlautopilot.core import Action, Env, Observation, Policy
from carlautopilot.data.episode_store import write_episode
from carlautopilot.scenario import ScenarioSpec

NOISE_AR = 0.9


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
) -> dict:
    """Drive with the expert and store (image, speed, command, expert action) per step.

    extra_samples(obs, action) may add (image, label) pairs for the same step, e.g. side cameras.

    The executed steering gets temporally correlated noise, but the stored label is the clean
    expert action, so the dataset covers off-centre states and shows how to recover from them.
    """
    obs = env.reset(scenario)
    if obs.image is None:
        raise ValueError("the environment must provide images (render_images=True)")
    expert.reset()
    rng = np.random.default_rng(noise_seed)
    limit = max_steps if max_steps is not None else int(scenario.max_time_s / env.dt) + 10
    noise = 0.0
    rows, images, target_speeds = [], [], []
    termination = None
    for step in range(limit):
        action = expert.act(obs).clipped()
        if step % stride == 0:
            rows.append((obs.t, obs.speed_mps, int(obs.command), action.steer, action.throttle, action.brake))
            images.append(obs.image)
            target_speeds.append(getattr(expert, "last_target_speed", None))
            for img, lab in extra_samples(obs, action) if extra_samples else ():
                rows.append((obs.t, obs.speed_mps, int(obs.command), lab.steer, lab.throttle, lab.brake))
                images.append(img)
                target_speeds.append(target_speeds[-1])
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
    stack = np.stack(images)
    meta = {
        "scenario_name": scenario.name,
        "seed": scenario.seed,
        "dt": env.dt,
        "image_height": int(stack.shape[1]),
        "image_width": int(stack.shape[2]),
    }
    write_episode(root, episode_id, meta, arrays, stack)
    return {"episode_id": episode_id, "steps": len(rows), "termination": termination}


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
