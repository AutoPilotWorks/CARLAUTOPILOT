from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from carlautopilot.carla_env.route import points_seen_from
from carlautopilot.core import Action, Env, Observation, Policy
from carlautopilot.data.episode_store import PERCEPT_KEYS, write_episode
from carlautopilot.models.route_steering import PERCEPT_RANGE_M
from carlautopilot.scenario import ScenarioSpec

NOISE_AR = 0.9
# DAgger: the expert takes the wheel past TAKEOVER_M off the route centre and hands it back under RETURN_M, so the
# model's own drift is recorded with expert labels without the episode ending off-road (OFF_ROAD_LATERAL_M = 2 m)
TAKEOVER_M, RETURN_M = 1.5, 0.3


def percept_labels(obs: Observation) -> tuple[float, ...] | None:
    """PERCEPT_KEYS from privileged observations: red light for my lane in range, its stop line m, nearest stop sign m
    (served or not), yield needed, junction entry m, lead gap m, lead relative speed m/s (0 without a lead); a distance
    at PERCEPT_RANGE_M means none in range."""
    if "lookahead_points" not in obs.extras:
        return None
    rng = PERCEPT_RANGE_M

    def near(key):
        d = obs.extras.get(key)
        return d if d is not None and d <= rng else None

    red = near("next_light_distance_m") if obs.extras.get("next_light_red") else None
    sign, yld = near("stop_sign_distance_m"), near("yield_distance_m")
    lead = obs.lead if obs.lead is not None and obs.lead.distance_m < rng else None
    return (float(red is not None), rng if red is None else red, rng if sign is None else sign,
            float(yld is not None), rng if yld is None else yld, lead.distance_m if lead else rng, lead.rel_speed_mps if lead else 0.0)


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
    png_compress: int = 6,
    driver: Policy | None = None,
) -> dict:
    """Drive with the expert and store (image, speed, command, expert action) per step, plus the step's light crop and
    surround camera images when the env gives them.

    With a driver (DAgger), the driver steers and the expert keeps the pedals and the labels: the data then covers
    the states the driver itself gets into (e.g. a model under-steering into a corner), not only the expert's.
    The expert takes over between TAKEOVER_M and RETURN_M of lateral error (needs privileged observations).

    The executed steering gets temporally correlated noise, but the stored label is the clean
    expert action, so the dataset covers off-centre states and shows how to recover from them.
    """
    t_start = time.perf_counter()
    obs = env.reset(scenario)
    if obs.image is None:
        raise ValueError("the environment must provide images (render_images=True)")
    expert.reset()
    if driver is not None:
        driver.reset()
    expert_on, takeovers, driver_steps = False, 0, 0
    rng = np.random.default_rng(noise_seed)
    limit = max_steps if max_steps is not None else int(scenario.max_time_s / env.dt) + 10
    noise = 0.0
    rows, images, target_speeds, percepts, lights, light_idx, views = [], [], [], [], [], [], []
    surround: dict[str, list[np.ndarray]] = {}  # the sedan rig's left/right/rear cameras, one image per step like light
    view_poses = getattr(env, "extra_views", ())
    labeler = copy.deepcopy(expert)  # steers the recovery views from their own pose; the expert's own state stays clean
    termination, collision = None, None
    for step in range(limit):
        action = expert.act(obs).clipped()
        if step % stride == 0:
            if "light_image" in obs.extras:
                lights.append(obs.extras["light_image"])
                for name, img in obs.extras.get("surround_images", {}).items():
                    surround.setdefault(name, []).append(img)
            # recovery views: the same step seen from a shifted/yawed camera, labelled with the expert's steer from that
            # pose (pedals, speed and perception labels are the front camera's); they share the step's light image
            shots = [(obs.image, action.steer)] + [
                (img, labeler.act(replace(obs, extras={**obs.extras, "lookahead_points": points_seen_from(obs.extras["lookahead_points"], dy, dyaw)})).clipped().steer)
                for (dy, dyaw), img in zip(view_poses, obs.extras.get("view_images", ()))
            ]
            for k, (img, steer_label) in enumerate(shots):
                rows.append((obs.t, obs.speed_mps, int(obs.command), steer_label, action.throttle, action.brake))
                images.append(img)
                target_speeds.append(getattr(expert, "last_target_speed", None))
                percepts.append(percept_labels(obs))
                light_idx.append(len(lights) - 1)
                views.append(k)
        steer = action.steer
        if driver is not None:
            lat = abs(obs.extras["lateral_error_m"])
            takeovers += lat > TAKEOVER_M and not expert_on
            expert_on = lat > TAKEOVER_M or (expert_on and lat > RETURN_M)
            driven = driver.act(obs)  # every step, so the driver's own timers (stop signs) stay current
            if not expert_on:
                steer, driver_steps = driven.steer, driver_steps + 1
        noise = NOISE_AR * noise + math.sqrt(1.0 - NOISE_AR**2) * float(rng.normal(0.0, steer_noise_std)) if steer_noise_std > 0 else 0.0
        res = env.step(Action(steer + noise, action.throttle, action.brake).clipped())
        obs = res.obs
        if res.done:
            termination = res.termination
            collision = next(([e.kind, e.detail] for e in res.events if e.kind.startswith("collision_")), None)
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
        for k, col in zip(PERCEPT_KEYS, zip(*percepts)):
            arrays[k] = np.array(col, dtype=np.float32)
    if view_poses:
        arrays["view"] = np.array(views, dtype=np.int64)
    step_images = {}
    if lights:
        arrays["light_idx"] = np.array(light_idx, dtype=np.int64)
        step_images = {"light": np.stack(lights)} | {k: np.stack(v) for k, v in surround.items()}
    stack = np.stack(images)
    meta = {
        "scenario_name": scenario.name,
        "seed": scenario.seed,
        "dt": env.dt,
        "image_height": int(stack.shape[1]),
        "image_width": int(stack.shape[2]),
    }
    t_sim = time.perf_counter()
    write_episode(root, episode_id, meta, arrays, stack, png_compress, step_images)
    t_end = time.perf_counter()
    info = {"episode_id": episode_id, "steps": len(rows), "termination": termination, "sim_s": round(t_sim - t_start, 1), "write_s": round(t_end - t_sim, 1)}
    if collision is not None:
        info["collision"] = collision  # [kind, other actor's type_id] (laptop's idea, 11-01)
    if driver is not None:
        info.update(takeovers=takeovers, driver_share=round(driver_steps / (step + 1), 3))
    return info


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
