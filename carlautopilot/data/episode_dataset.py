from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from carlautopilot.data.episode_store import PERCEPTION_KEYS, SPLIT_PERCEPTION_KEYS, load_image, read_episode


def build_image_cache(root: str | Path, episode_id: str, cache_dir: str | Path, n: int, folder: str = "images") -> Path:
    """Decode an episode's PNGs once into a uint8 (N, H, W, 3) .npy file; pixels are identical to the PNGs."""
    path = Path(cache_dir) / (f"{episode_id}.npy" if folder == "images" else f"{episode_id}_{folder}.npy")
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    first = load_image(root, episode_id, 0, folder)
    tmp = path.with_name(path.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n,) + first.shape)
    for i in range(n):
        arr[i] = load_image(root, episode_id, i, folder)
    arr.flush()
    del arr
    tmp.replace(path)
    return path


class EpisodeDataset(Dataset):
    def __init__(self, root: str | Path, episode_ids: list[str], cache_dir: str | Path | None = None, target_speed: bool = False, perception: bool = False, split_stops: bool = False, light: bool = False):
        """target_speed appends the expert's road-geometry speed label as a 4th target column; perception (with it)
        appends PERCEPTION_KEYS as columns 5-8."""
        if not episode_ids:
            raise ValueError("episode_ids is empty")
        self.root = Path(root)
        self.target_speed = target_speed
        self.perception = perception
        self.percept_keys = SPLIT_PERCEPTION_KEYS if split_stops else PERCEPTION_KEYS
        self.light = light
        self._light_cache: dict[str, Path] = {}
        self._cache: dict[str, Path] = {}
        self._mmaps: dict[str, np.ndarray] = {}
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._index: list[tuple[str, int]] = []
        for eid in episode_ids:
            _, arrays = read_episode(self.root, eid)
            if target_speed and "target_speed" not in arrays:
                raise ValueError(f"{eid} has no target_speed labels")
            if perception and any(k not in arrays for k in self.percept_keys):
                raise ValueError(f"{eid} has no perception labels")
            if light and "light_idx" not in arrays:
                raise ValueError(f"{eid} has no traffic-light camera images")
            self._arrays[eid] = arrays
            self._index.extend((eid, i) for i in range(len(arrays["t"])))
            if cache_dir is not None:
                self._cache[eid] = build_image_cache(self.root, eid, cache_dir, len(arrays["t"]))
                if light:
                    self._light_cache[eid] = build_image_cache(self.root, eid, cache_dir, int(arrays["light_idx"].max()) + 1, "light")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        eid, i = self._index[idx]
        a = self._arrays[eid]
        if eid in self._cache:
            if eid not in self._mmaps:
                self._mmaps[eid] = np.load(self._cache[eid], mmap_mode="r")
            pixels = np.array(self._mmaps[eid][i])
        else:
            pixels = load_image(self.root, eid, i)
        image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        cols = [a["steer"][i], a["throttle"][i], a["brake"][i]] + ([a["target_speed"][i]] if self.target_speed else [])
        cols += [a[k][i] for k in self.percept_keys] if self.perception else []
        target = np.array(cols, dtype=np.float32)
        item = {}
        if self.light:
            j = int(a["light_idx"][i])
            if eid in self._light_cache:
                key = eid + "/light"
                if key not in self._mmaps:
                    self._mmaps[key] = np.load(self._light_cache[eid], mmap_mode="r")
                lp = np.array(self._mmaps[key][j])
            else:
                lp = load_image(self.root, eid, j, "light")
            item["light"] = torch.from_numpy(lp).permute(2, 0, 1).float() / 255.0
        return item | {
            "image": image,
            "speed": torch.tensor(float(a["speed"][i]), dtype=torch.float32),
            "command": torch.tensor(int(a["command"][i]), dtype=torch.int64),
            "target": torch.from_numpy(target),
        }
