from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from carlautopilot.data.episode_store import load_image, read_episode


class EpisodeDataset(Dataset):
    def __init__(self, root: str | Path, episode_ids: list[str]):
        if not episode_ids:
            raise ValueError("episode_ids is empty")
        self.root = Path(root)
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._index: list[tuple[str, int]] = []
        for eid in episode_ids:
            _, arrays = read_episode(self.root, eid)
            self._arrays[eid] = arrays
            self._index.extend((eid, i) for i in range(len(arrays["t"])))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        eid, i = self._index[idx]
        a = self._arrays[eid]
        image = torch.from_numpy(load_image(self.root, eid, i)).permute(2, 0, 1).float() / 255.0
        target = np.array([a["steer"][i], a["throttle"][i], a["brake"][i]], dtype=np.float32)
        return {
            "image": image,
            "speed": torch.tensor(float(a["speed"][i]), dtype=torch.float32),
            "command": torch.tensor(int(a["command"][i]), dtype=torch.int64),
            "target": torch.from_numpy(target),
        }
