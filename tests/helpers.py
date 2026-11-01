import numpy as np

from carlautopilot.core import Command
from carlautopilot.data.episode_store import write_episode

STEER_BY_COMMAND = {Command.FOLLOW: 0.0, Command.LEFT: -0.5, Command.RIGHT: 0.5, Command.STRAIGHT: 0.0}


def make_episodes(root, n_episodes=4, steps=20, h=32, w=64, seed=0, target_speed=False, perception=False, light=False, surround=False):
    rng = np.random.default_rng(seed)
    ids = []
    for e in range(n_episodes):
        eid = f"ep{e:03d}"
        command = rng.integers(0, 4, size=steps)
        steer = np.array([STEER_BY_COMMAND[Command(int(c))] for c in command], dtype=np.float32)
        arrays = {
            "t": np.arange(steps) * 0.1,
            "speed": rng.uniform(0, 10, size=steps).astype(np.float32),
            "command": command.astype(np.int64),
            "steer": steer,
            "throttle": np.full(steps, 0.3, dtype=np.float32),
            "brake": np.zeros(steps, dtype=np.float32),
        }
        if target_speed:
            arrays["target_speed"] = np.where(command == 0, 8.0, 4.0).astype(np.float32)
        if perception:
            arrays["lead_dist"] = rng.choice([12.0, 40.0], size=steps).astype(np.float32)
            arrays["lead_rel"] = np.where(arrays["lead_dist"] < 40, -1.0, 0.0).astype(np.float32)
            arrays["red"] = (rng.random(steps) < 0.3).astype(np.float32)
            arrays["red_dist"] = np.where(arrays["red"] > 0, 20.0, 40.0).astype(np.float32)
            arrays["sign_dist"] = rng.choice([8.0, 40.0], size=steps).astype(np.float32)
            arrays["yield"] = (rng.random(steps) < 0.2).astype(np.float32)
            arrays["yield_dist"] = np.where(arrays["yield"] > 0, 10.0, 40.0).astype(np.float32)
        images = rng.integers(0, 256, size=(steps, h, w, 3), dtype=np.uint8)
        light_images = None
        if light:
            arrays["light_idx"] = (np.arange(steps) // 2).astype(np.int64)  # two view rows per time step
            light_images = rng.integers(0, 256, size=(int(arrays["light_idx"][-1]) + 1, 16, 32, 3), dtype=np.uint8)
        meta = {"scenario_name": "synthetic", "seed": seed, "dt": 0.1, "image_height": h, "image_width": w}
        step_images = {"light": light_images} if light else {}
        if surround:  # the sedan rig's left/right/rear, same size as the main image
            step_images |= {f: rng.integers(0, 256, size=(len(light_images), h, w, 3), dtype=np.uint8) for f in ("left", "right", "rear")}
        write_episode(root, eid, meta, arrays, images, step_images=step_images or None)
        ids.append(eid)
    return ids
