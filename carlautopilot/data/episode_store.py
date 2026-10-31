from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ARRAY_KEYS = ("t", "speed", "command", "steer", "throttle", "brake")
PERCEPTION_KEYS = ("light_red", "light_dist", "lead_dist", "lead_rel")
# the merged "stop needed" above split by cause: red light (+ stop line), any stop sign ahead, yield (+ junction entry)
STOP_KEYS = ("red", "red_dist", "sign_dist", "yield", "yield_dist")
SPLIT_PERCEPTION_KEYS = STOP_KEYS + ("lead_dist", "lead_rel")
# light_idx: row -> time-step index into the per-step camera folders (one image per time step, shared by that step's
# view rows): light/ always, plus the sedan's surround cameras left/ right/ rear/ when it has them
OPTIONAL_KEYS = ("target_speed",) + PERCEPTION_KEYS + STOP_KEYS + ("light_idx",)
STEP_FOLDERS = ("light", "left", "right", "rear")
META_KEYS = ("scenario_name", "seed", "dt", "image_height", "image_width")


def episode_dir(root: str | Path, episode_id: str) -> Path:
    return Path(root) / episode_id


def write_episode(root: str | Path, episode_id: str, meta: dict, arrays: dict[str, np.ndarray], images: np.ndarray, png_compress: int = 6,
                  step_images: dict[str, np.ndarray] | None = None) -> Path:
    """png_compress is the zlib level 0-9 (6 = Pillow default); PNG is lossless, so it only trades file size for speed.
    step_images: STEP_FOLDERS name -> (steps, H, W, 3), indexed by arrays['light_idx']; "light" is required with any."""
    step_images = step_images or {}
    d = episode_dir(root, episode_id)
    if d.exists():
        raise FileExistsError(f"episode already exists: {d}")
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"meta is missing keys: {missing}")
    missing = [k for k in ARRAY_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"arrays are missing keys: {missing}")
    n = len(arrays["t"])
    keys = ARRAY_KEYS + tuple(k for k in OPTIONAL_KEYS if k in arrays)
    if any(len(arrays[k]) != n for k in keys) or len(images) != n:
        raise ValueError("array and image lengths differ")
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[3] != 3:
        raise ValueError("images must be uint8 with shape (N, H, W, 3)")
    if images.shape[1] != meta["image_height"] or images.shape[2] != meta["image_width"]:
        raise ValueError("image size does not match meta")
    if set(step_images) - set(STEP_FOLDERS):
        raise ValueError(f"unknown step image folders {sorted(set(step_images) - set(STEP_FOLDERS))}")
    if ("light" in step_images) != ("light_idx" in arrays) or (step_images and "light" not in step_images):
        raise ValueError("step images need the light images and arrays['light_idx'], and those come together")
    if any(int(arrays["light_idx"].max()) + 1 != len(v) for v in step_images.values()):
        raise ValueError("every step image folder needs one image per light_idx step")
    (d / "images").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(d / "steps.npz", **{k: np.asarray(arrays[k]) for k in keys})
    for i, img in enumerate(images):
        Image.fromarray(img).save(d / "images" / f"{i:06d}.png", compress_level=png_compress)
    for folder, stack in step_images.items():
        (d / folder).mkdir()
        for i, img in enumerate(stack):
            Image.fromarray(img).save(d / folder / f"{i:06d}.png", compress_level=png_compress)
    return d


def validate_episode(root: str | Path, episode_id: str) -> int:
    d = episode_dir(root, episode_id)
    if not (d / "meta.json").is_file():
        raise ValueError(f"{episode_id}: missing meta.json")
    if not (d / "steps.npz").is_file():
        raise ValueError(f"{episode_id}: missing steps.npz")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{episode_id}: meta is missing keys {missing}")
    with np.load(d / "steps.npz") as z:
        missing = [k for k in ARRAY_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{episode_id}: steps.npz is missing arrays {missing}")
        lengths = {k: len(z[k]) for k in ARRAY_KEYS}
        n_light = int(z["light_idx"].max()) + 1 if "light_idx" in z.files else 0
    n = lengths["t"]
    if any(v != n for v in lengths.values()):
        raise ValueError(f"{episode_id}: array length mismatch {lengths}")
    files = sorted(p.name for p in (d / "images").glob("*.png")) if (d / "images").is_dir() else []
    expected = [f"{i:06d}.png" for i in range(n)]
    if files != expected:
        raise ValueError(f"{episode_id}: image files do not match steps (found {len(files)}, expected {n})")
    with Image.open(d / "images" / expected[0]) as im:
        if im.size != (meta["image_width"], meta["image_height"]):
            raise ValueError(f"{episode_id}: image size {im.size} differs from meta")
    for folder in STEP_FOLDERS:
        if (folder == "light" and n_light) or (d / folder).is_dir():
            if sorted(p.name for p in (d / folder).glob("*.png")) != [f"{i:06d}.png" for i in range(n_light)]:
                raise ValueError(f"{episode_id}: {folder} camera files do not match light_idx")
    return n


def read_episode(root: str | Path, episode_id: str) -> tuple[dict, dict[str, np.ndarray]]:
    validate_episode(root, episode_id)
    d = episode_dir(root, episode_id)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    with np.load(d / "steps.npz") as z:
        arrays = {k: z[k] for k in ARRAY_KEYS + OPTIONAL_KEYS if k in z.files}
    return meta, arrays


def load_image(root: str | Path, episode_id: str, index: int, folder: str = "images") -> np.ndarray:
    with Image.open(episode_dir(root, episode_id) / folder / f"{index:06d}.png") as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def list_episodes(root: str | Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir() and (p / "meta.json").is_file())
