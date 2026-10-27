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
from carlautopilot.carla_env.env import VEHICLES, CarlaEnv, switch_map
from carlautopilot.control import IDM, ACCController
from carlautopilot.core import Action, Observation
from carlautopilot.data.collect import collect_episode
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.scenario import ScenarioSpec

ACC = ACCController(IDM(s0=4.0))  # gaps are measured along the route, so keep a larger standstill margin


def make_scenarios(env: CarlaEnv, a: argparse.Namespace) -> list[ScenarioSpec]:
    rng = random.Random(f"{a.seed}-{a.town}")
    return [
        env.make_route_scenario(f"{a.town}_{a.seed}_{i:03d}", rng, a.carla_root, npc_vehicles=a.npc, speed_limit_mps=a.speed, npc_walkers=a.walkers)
        for i in range(a.episodes)
    ]


# pure-pursuit look-ahead (gain, offset m, min m); the sedan's short one stops junction corner cutting (2026-10-23)
EXPERT_LOOKAHEAD = {"sedan": (0.5, 2.0, 3.0), "truck": (0.5, 2.0, 3.0), "truck_cab": (0.5, 2.0, 3.0)}


def expert(speed: float, vehicle: str = "sedan") -> PathFollowPolicy:
    _, wheelbase, max_steer, _, _, _ = VEHICLES[vehicle]
    gain, offset, minimum = EXPERT_LOOKAHEAD[vehicle]
    return PathFollowPolicy(speed_limit_mps=speed, wheelbase_m=wheelbase, max_steer_rad=max_steer, acc=ACC, max_lat_accel=2.0,
                            lookahead_gain=gain, lookahead_offset_m=offset, lookahead_min_m=minimum)


def view_labeler(views: tuple[tuple[float, float], ...], speed: float, vehicle: str = "sedan"):
    """Label each extra camera with the expert steer for the pose that camera represents; pedals stay the centre ones."""
    pol = expert(speed, vehicle)

    def samples(obs: Observation, action: Action) -> list:
        out = []
        for (dy, dyaw), img in zip(views, obs.extras["view_images"]):
            shifted = replace(obs, extras={**obs.extras, "lookahead_points": points_seen_from(obs.extras["lookahead_points"], dy, dyaw)})
            out.append((img, Action(pol.act(shifted).steer, action.throttle, action.brake)))
        return out

    return samples


def parse_views(spec: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(dy), float(dyaw)) for dy, dyaw in (v.split(":") for v in spec.split(",") if v))


def parse_rigs(spec: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(f), float(t)) for f, t in (v.split(":") for v in spec.split(",") if v))


def rig_roots(data_root: str, rigs_spec: str) -> list[Path]:
    """Without --rigs the data goes to --data-root; with it, each rig gets <data-root>_fov<F>_turn<T>."""
    if not rigs_spec:
        return [Path(data_root)]
    return [Path(f"{data_root}_fov{f:g}_turn{t:g}") for f, t in parse_rigs(rigs_spec)]


def collect(a: argparse.Namespace, env: CarlaEnv) -> None:
    roots = rig_roots(a.data_root, a.rigs)
    root = roots[0]
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    for sc in make_scenarios(env, a):
        eid = sc.name
        if (root / eid).exists():
            continue
        sc.save(root / "scenarios" / f"{eid}.json")
        t0 = time.time()
        info = collect_episode(
            env, expert(a.speed, a.vehicle), sc, root, eid, steer_noise_std=a.noise, noise_seed=zlib.crc32(eid.encode()), stride=a.stride,
            extra_samples=view_labeler(env.extra_views, a.speed, a.vehicle) if env.extra_views else None, png_compress=a.png_compress,
            stream_images=(lambda obs: obs.extras["rig_images"]) if len(roots) > 1 else None, stream_roots=roots[1:],
        )
        info["wall_s"] = round(time.time() - t0, 1)
        info["weather"] = sc.meta["weather"]
        with open(root / "collect_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
        print(json.dumps(info), flush=True)


def evaluate(a: argparse.Namespace, env: CarlaEnv) -> None:
    from carlautopilot.runner import run_suite

    scs = make_scenarios(env, a)
    if a.checkpoint is None:
        factory, title = (lambda: expert(a.speed, a.vehicle)), f"CARLA {a.town}: privileged expert ({a.vehicle})"
    else:
        from carlautopilot.policies.model_policy import ModelPolicy

        factory = lambda: ModelPolicy.from_checkpoint(a.checkpoint, speed_limit_mps=a.model_speed, use_model_pedals=a.model_pedals, acc=ACC, turn_speed_mps=a.turn_speed, device=a.device, sim_lead=a.sim_lead)
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
        q.add_argument("--walkers", type=int, default=0, help="pedestrians with CARLA's walker AI (some cross roads)")
        q.add_argument("--image-size", default="128x256", help="camera image HxW")
        q.add_argument("--junction-rules", action="store_true", help="stop signs (served + STOP_SIGN events) and junction yielding")
        q.add_argument("--speed", type=float, default=8.0)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--host", default="localhost")
        q.add_argument("--port", type=int, default=2000)
        q.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port; give each parallel CARLA server its own")
        q.add_argument("--vehicle", default="sedan", choices=sorted(VEHICLES))
    c = sub.choices["collect"]
    c.add_argument("--data-root", required=True)
    c.add_argument("--noise", type=float, default=0.05)
    c.add_argument("--stride", type=int, default=2)
    c.add_argument("--png-compress", type=int, default=6, help="PNG zlib level 0-9 (lossless); lower is faster, files larger")
    c.add_argument("--rigs", default="", help="camera rigs as fov:turn_deg, comma separated, e.g. 120:0,90:25 (one data root per rig)")
    c.add_argument("--views", default="", help="extra cameras as left_offset_m:left_yaw_deg, comma separated, e.g. 1:0,-1:0,0:10,0:-10")
    e = sub.choices["eval"]
    e.add_argument("--out-dir", required=True)
    e.add_argument("--checkpoint")
    e.add_argument("--model-pedals", action="store_true")
    e.add_argument("--model-speed", type=float, default=6.0, help="ACC speed limit for the model (expert data rarely exceeds ~5-6 m/s); with a predict_speed checkpoint, the road limit above the learned speed")
    e.add_argument("--turn-speed", type=float, default=4.5, help="ACC speed cap while the command is LEFT/RIGHT/STRAIGHT (rule; ignored by predict_speed checkpoints)")
    e.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    e.add_argument("--sim-lead", action="store_true", help="ablation: a perceiving model still gets the simulator's lead vehicle")
    e.add_argument("--fov", type=float, default=90.0)
    e.add_argument("--turn-camera", type=float, default=0.0, help="swing the camera toward LEFT/RIGHT commands by this many degrees")
    a = p.parse_args(argv)
    switch_map(a.town, a.host, a.port)
    views = parse_views(a.views) if a.mode == "collect" else ()
    rigs = (parse_rigs(a.rigs) or ((90.0, 0.0),)) if a.mode == "collect" else ((a.fov, a.turn_camera),)
    h, w = (int(v) for v in a.image_size.lower().split("x"))
    env = CarlaEnv(a.host, a.port, tm_port=a.tm_port, image_size=(h, w), junction_rules=a.junction_rules, privileged_obs=a.mode == "collect" or a.checkpoint is None, extra_views=views, vehicle=a.vehicle, rigs=rigs)
    try:
        collect(a, env) if a.mode == "collect" else evaluate(a, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
