"""CARLA data collection with the privileged expert, and closed-loop evaluation of the expert or a checkpoint.

Needs a running CARLA 0.9.16 server. One town per run (the map is switched from a separate process).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from dataclasses import replace
from collections import Counter
from pathlib import Path

from carlautopilot.carla_env.route import points_seen_from
from carlautopilot.carla_env.env import MAX_STEER_RAD, WHEELBASE_M, CarlaEnv, switch_map
from carlautopilot.control import IDM, ACCController
from carlautopilot.core import Action, Observation
from carlautopilot.data.collect import collect_episode
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.scenario import ScenarioSpec

ACC = ACCController(IDM(s0=4.0))  # gaps are measured along the route, so keep a larger standstill margin


def make_scenarios(env: CarlaEnv, a: argparse.Namespace) -> list[ScenarioSpec]:
    rng = random.Random(f"{a.seed}-{a.town}")
    return [
        env.make_route_scenario(f"{a.town}_{a.seed}_{i:03d}", rng, a.carla_root, npc_vehicles=a.npc, speed_limit_mps=a.speed)
        for i in range(a.episodes)
    ]


def expert(speed: float) -> PathFollowPolicy:
    return PathFollowPolicy(speed_limit_mps=speed, wheelbase_m=WHEELBASE_M, max_steer_rad=MAX_STEER_RAD, acc=ACC, max_lat_accel=2.0,
        lookahead_gain=0.5, lookahead_offset_m=2.0, lookahead_min_m=3.0)  # shorter look-ahead: long one cut junction corners


def view_labeler(views: tuple[tuple[float, float], ...], speed: float):
    """Label each extra camera with the expert steer for the pose that camera represents; pedals stay the centre ones."""
    pol = expert(speed)

    def samples(obs: Observation, action: Action) -> list:
        out = []
        for (dy, dyaw), img in zip(views, obs.extras["view_images"]):
            shifted = replace(obs, extras={**obs.extras, "lookahead_points": points_seen_from(obs.extras["lookahead_points"], dy, dyaw)})
            out.append((img, Action(pol.act(shifted).steer, action.throttle, action.brake)))
        return out

    return samples


def parse_views(spec: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(dy), float(dyaw)) for dy, dyaw in (v.split(":") for v in spec.split(",") if v))


def collect(a: argparse.Namespace, env: CarlaEnv) -> None:
    root = Path(a.data_root)
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    for sc in make_scenarios(env, a):
        eid = sc.name
        if (root / eid).exists():
            continue
        sc.save(root / "scenarios" / f"{eid}.json")
        t0 = time.time()
        info = collect_episode(env, expert(a.speed), sc, root, eid, steer_noise_std=a.noise, noise_seed=zlib.crc32(eid.encode()), stride=a.stride, extra_samples=view_labeler(env.extra_views, a.speed) if env.extra_views else None)
        info["wall_s"] = round(time.time() - t0, 1)
        info["weather"] = sc.meta["weather"]
        with open(root / "collect_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
        print(json.dumps(info), flush=True)


def evaluate(a: argparse.Namespace, env: CarlaEnv) -> None:
    from carlautopilot.runner import run_suite

    scs = make_scenarios(env, a)
    if a.checkpoint is None:
        factory, title = (lambda: expert(a.speed)), "CARLA: privileged expert"
    else:
        from carlautopilot.policies.model_policy import ModelPolicy

        factory = lambda: ModelPolicy.from_checkpoint(a.checkpoint, speed_limit_mps=a.model_speed, use_model_pedals=a.model_pedals, acc=ACC, turn_speed_mps=a.turn_speed, device=a.device)
        title = f"CARLA {a.town}: model {'steer+pedals' if a.model_pedals else 'steer, ACC pedals'}"
    res = run_suite(lambda: env, factory, scs, a.out_dir, title=title)
    print(json.dumps(res["aggregate"], ensure_ascii=False))
    print(Counter(e["termination"] for e in res["episodes"]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("collect", "eval"):
        q = sub.add_parser(name)
        q.add_argument("--carla-root", required=True, help="CARLA package folder (for PythonAPI/carla/agents)")
        q.add_argument("--town", default="Town10HD_Opt")
        q.add_argument("--episodes", type=int, default=10)
        q.add_argument("--npc", type=int, default=30)
        q.add_argument("--speed", type=float, default=8.0)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--host", default="localhost")
        q.add_argument("--port", type=int, default=2000)
    c = sub.choices["collect"]
    c.add_argument("--data-root", required=True)
    c.add_argument("--noise", type=float, default=0.05)
    c.add_argument("--stride", type=int, default=2)
    c.add_argument("--views", default="", help="extra cameras as left_offset_m:left_yaw_deg, comma separated, e.g. 1:0,-1:0,0:10,0:-10")
    e = sub.choices["eval"]
    e.add_argument("--out-dir", required=True)
    e.add_argument("--checkpoint")
    e.add_argument("--model-pedals", action="store_true")
    e.add_argument("--model-speed", type=float, default=6.0, help="ACC speed limit for the model (expert data rarely exceeds ~5-6 m/s)")
    e.add_argument("--turn-speed", type=float, default=4.5, help="ACC speed cap while the command is LEFT/RIGHT/STRAIGHT (rule)")
    e.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    a = p.parse_args(argv)
    switch_map(a.town, a.host, a.port)
    views = parse_views(a.views) if a.mode == "collect" else ()
    env = CarlaEnv(a.host, a.port, privileged_obs=a.mode == "collect" or a.checkpoint is None, extra_views=views)
    try:
        collect(a, env) if a.mode == "collect" else evaluate(a, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
