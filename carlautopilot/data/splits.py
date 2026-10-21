from __future__ import annotations

import random


def make_folds(episode_ids: list[str], k: int, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    ids = sorted(set(episode_ids))
    if len(ids) != len(episode_ids):
        raise ValueError("duplicate episode ids")
    if k < 2 or k > len(ids):
        raise ValueError("k must be between 2 and the number of episodes")
    random.Random(seed).shuffle(ids)
    base, extra = divmod(len(ids), k)
    chunks, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunks.append(ids[start : start + size])
        start += size
    return [(sorted(x for j, c in enumerate(chunks) if j != i for x in c), sorted(chunks[i])) for i in range(k)]
