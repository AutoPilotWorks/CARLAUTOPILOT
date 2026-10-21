import numpy as np

from carlautopilot.core import Command
from carlautopilot.data.episode_store import write_episode

STEER_BY_COMMAND = {Command.FOLLOW: 0.0, Command.LEFT: -0.5, Command.RIGHT: 0.5, Command.STRAIGHT: 0.0}


def make_episodes(root, n_episodes=4, steps=20, h=32, w=64, seed=0):
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
        images = rng.integers(0, 256, size=(steps, h, w, 3), dtype=np.uint8)
        meta = {"scenario_name": "synthetic", "seed": seed, "dt": 0.1, "image_height": h, "image_width": w}
        write_episode(root, eid, meta, arrays, images)
        ids.append(eid)
    return ids
