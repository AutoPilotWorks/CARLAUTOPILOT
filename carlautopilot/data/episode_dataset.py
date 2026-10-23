from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from carlautopilot.data.episode_store import load_image, read_episode


def build_image_cache(root: str | Path, episode_id: str, cache_dir: str | Path, n: int) -> Path:
    """Decode an episode's PNGs once into a uint8 (N, H, W, 3) .npy file; pixels are identical to the PNGs."""
    path = Path(cache_dir) / f"{episode_id}.npy"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    first = load_image(root, episode_id, 0)
    tmp = path.with_name(path.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n,) + first.shape)
    for i in range(n):
        arr[i] = load_image(root, episode_id, i)
    arr.flush()
    del arr
    tmp.replace(path)
    return path


class EpisodeDataset(Dataset):
    def __init__(self, root: str | Path, episode_ids: list[str], cache_dir: str | Path | None = None):
        if not episode_ids:
            raise ValueError("episode_ids is empty")
        self.root = Path(root)
        self._cache: dict[str, Path] = {}
        self._mmaps: dict[str, np.ndarray] = {}
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._index: list[tuple[str, int]] = []
        for eid in episode_ids:
            _, arrays = read_episode(self.root, eid)
            self._arrays[eid] = arrays
            self._index.extend((eid, i) for i in range(len(arrays["t"])))
            if cache_dir is not None:
                self._cache[eid] = build_image_cache(self.root, eid, cache_dir, len(arrays["t"]))

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
        target = np.array([a["steer"][i], a["throttle"][i], a["brake"][i]], dtype=np.float32)
        return {
            "image": image,
            "speed": torch.tensor(float(a["speed"][i]), dtype=torch.float32),
            "command": torch.tensor(int(a["command"][i]), dtype=torch.int64),
            "target": torch.from_numpy(target),
        }
