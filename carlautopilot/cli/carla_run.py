"""CARLA data collection with the privileged expert, and closed-loop evaluation of the expert or a checkpoint.

Needs a running CARLA 0.9.16 server. One town per run (the map is switched from a separate process).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from collections import Counter
from pathlib import Path

from carlautopilot.carla_env.env import MAX_STEER_RAD, WHEELBASE_M, CarlaEnv, switch_map
from carlautopilot.control import IDM, ACCController
from carlautopilot.data.collect import collect_episode
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.scenario import ScenarioSpec

ACC = ACCController(IDM(s0=4.0))  # gaps are measured along the route, so keep a larger standstill margin
# a waypoints model follows its predicted path with the expert's pure pursuit (car geometry and look-ahead)
MODEL_STEER = dict(wheelbase_m=WHEELBASE_M, max_steer_rad=MAX_STEER_RAD, lookahead_gain=0.5, lookahead_offset_m=2.0, lookahead_min_m=3.0)


def make_scenarios(env: CarlaEnv, a: argparse.Namespace) -> list[ScenarioSpec]:
    rng = random.Random(f"{a.seed}-{a.town}")
    return [
        env.make_route_scenario(f"{a.town}_{a.seed}_{i:03d}", rng, a.carla_root, npc_vehicles=a.npc, speed_limit_mps=a.speed, npc_walkers=a.walkers)
        for i in range(a.episodes)
    ]


def expert(speed: float, start_delay_max_s: float = 0.0, start_delay_seed: int = 0, light_stop_m: float | None = None) -> PathFollowPolicy:
    """light_stop_m: halt this far before a red light's line (default: LIGHT_STOP_MARGIN_M + the IDM standstill gap)."""
    margin = {} if light_stop_m is None else {"light_stop_margin_m": light_stop_m - ACC.idm.s0}
    # the short pure-pursuit look-ahead stops junction corner cutting (2026-10-23)
    return PathFollowPolicy(speed_limit_mps=speed, wheelbase_m=WHEELBASE_M, max_steer_rad=MAX_STEER_RAD, acc=ACC, max_lat_accel=2.0,
                            lookahead_gain=0.5, lookahead_offset_m=2.0, lookahead_min_m=3.0,
                            start_delay_max_s=start_delay_max_s, start_delay_seed=start_delay_seed, **margin)


def parse_pairs(spec: str) -> tuple[tuple[float, float], ...]:
    """'a:b,c:d' -> ((a, b), (c, d))"""
    return tuple((float(p), float(q)) for p, q in (v.split(":") for v in spec.split(",") if v))


def collect(a: argparse.Namespace, env: CarlaEnv) -> None:
    root = Path(a.data_root)
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    driver = None
    if a.dagger:
        from carlautopilot.policies.model_policy import ModelPolicy

        driver = ModelPolicy.from_checkpoint(a.dagger, acc=ACC, device="cuda", **MODEL_STEER)  # only its steering is used
    for sc in make_scenarios(env, a):
        eid = sc.name
        if (root / eid).exists():
            continue
        sc.save(root / "scenarios" / f"{eid}.json")
        t0 = time.time()
        stop_m = None
        if a.light_stop:
            (lo, hi), = parse_pairs(a.light_stop)
            stop_m = round(random.Random(zlib.crc32(eid.encode())).uniform(lo, hi), 2)
        info = collect_episode(env, expert(a.speed, a.start_delay, zlib.crc32(eid.encode()), stop_m), sc, root, eid, steer_noise_std=a.noise,
                               noise_seed=zlib.crc32(eid.encode()), stride=a.stride, png_compress=a.png_compress, driver=driver)
        info["wall_s"] = round(time.time() - t0, 1)
        if stop_m is not None:
            info["light_stop_m"] = stop_m
        info["weather"] = sc.meta["weather"]
        if a.dagger:
            info["driver"] = a.dagger
        with open(root / "collect_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
        print(json.dumps(info), flush=True)


def evaluate(a: argparse.Namespace, env: CarlaEnv) -> None:
    from carlautopilot.runner import run_suite

    scs = make_scenarios(env, a)
    if a.checkpoint is None:
        factory, title = (lambda: expert(a.speed)), f"CARLA {a.town}: privileged expert"
    else:
        from carlautopilot.policies.model_policy import ModelPolicy

        factory = lambda: ModelPolicy.from_checkpoint(a.checkpoint, speed_limit_mps=a.model_speed, acc=ACC, device=a.device, **MODEL_STEER)
        title = f"CARLA {a.town}: model"
    view = None
    if a.view:
        from carlautopilot.eval.live_view import LiveView

        view = LiveView(f"{a.town} - {'expert' if a.checkpoint is None else Path(a.checkpoint).parent.name}")
    res = run_suite(lambda: env, factory, scs, a.out_dir, title=title, event_frames=a.event_frames, view=view)
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
        q.add_argument("--image-size", default="160x320", help="driving image HxW (the front camera renders at 4x this)")
        q.add_argument("--fov", type=float, default=120.0, help="front camera field of view (deg)")
        q.add_argument("--junction-rules", action="store_true", help="stop signs (served + STOP_SIGN events) and junction yielding")
        q.add_argument("--speed", type=float, default=8.0)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--host", default="localhost")
        q.add_argument("--port", type=int, default=2000)
        q.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port; give each parallel CARLA server its own")
        q.add_argument("--walker-cross", type=float, default=0.2, help="share of pedestrians that cross roads")
        q.add_argument("--path-walkers", type=int, default=0, help="scripted pedestrians per episode along the lane edge / crossing ahead")
        q.add_argument("--lead-brakes", type=int, default=0, help="cars put ahead on the route that brake to a stop now and then")
        q.add_argument("--radar", action="store_true", help="front radar; a perceiving model fuses its in-path obstacle with the camera's")
    c = sub.choices["collect"]
    c.add_argument("--data-root", required=True)
    c.add_argument("--noise", type=float, default=0.05)
    c.add_argument("--stride", type=int, default=4)
    c.add_argument("--start-delay", type=float, default=0.0, help="expert waits a random 0..this s after a stop reason clears (reaction time)")
    c.add_argument("--png-compress", type=int, default=6, help="PNG zlib level 0-9 (lossless); lower is faster, files larger")
    c.add_argument("--dagger", help="checkpoint that steers while the expert labels (DAgger)")
    c.add_argument("--views", default="", help="recovery cameras as left_offset_m:left_yaw_deg, comma separated, e.g. 1:0,-1:0,0:10,0:-10")
    c.add_argument("--light-stop", default="", help="lo:hi m: the expert halts a uniform random lo..hi m before red-light lines (per episode)")
    e = sub.choices["eval"]
    e.add_argument("--out-dir", required=True)
    e.add_argument("--checkpoint")
    e.add_argument("--event-frames", action="store_true", help="save the last 4 s of camera frames at every event / failed ending")
    e.add_argument("--view", action="store_true", help="live window: the cameras the model sees and what it decides (display only)")
    e.add_argument("--model-speed", type=float, default=6.0, help="ACC road speed limit for the model (m/s); a predict_speed checkpoint drives at min(this, learned speed)")
    e.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    a = p.parse_args(argv)
    switch_map(a.town, a.host, a.port)
    h, w = (int(v) for v in a.image_size.lower().split("x"))
    env = CarlaEnv(a.host, a.port, tm_port=a.tm_port, image_size=(h, w), fov=a.fov, junction_rules=a.junction_rules,
                   privileged_obs=a.mode == "collect" or a.checkpoint is None, radar=a.radar, walker_cross=a.walker_cross, path_walkers=a.path_walkers, lead_brakes=a.lead_brakes,
                   extra_views=parse_pairs(a.views) if a.mode == "collect" else ())
    try:
        collect(a, env) if a.mode == "collect" else evaluate(a, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
