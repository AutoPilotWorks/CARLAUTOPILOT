import json

import numpy as np
import pytest
import torch

from carlautopilot.data.episode_dataset import EpisodeDataset
from carlautopilot.data.episode_store import list_episodes, load_image, read_episode, validate_episode, write_episode
from carlautopilot.data.splits import make_folds
from tests.helpers import make_episodes


def test_store_roundtrip(tmp_path):
    ids = make_episodes(tmp_path, n_episodes=2, steps=5)
    assert list_episodes(tmp_path) == ids
    meta, arrays = read_episode(tmp_path, ids[0])
    assert meta["image_height"] == 32 and len(arrays["t"]) == 5
    assert validate_episode(tmp_path, ids[0]) == 5
    img = load_image(tmp_path, ids[0], 3)
    assert img.shape == (32, 64, 3) and img.dtype == np.uint8


def test_image_roundtrip_is_lossless(tmp_path):
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(3, 8, 8, 3), dtype=np.uint8)
    arrays = {k: np.zeros(3, dtype=np.float32) for k in ("t", "speed", "command", "steer", "throttle", "brake")}
    meta = {"scenario_name": "s", "seed": 0, "dt": 0.1, "image_height": 8, "image_width": 8}
    write_episode(tmp_path, "e", meta, arrays, images)
    for i in range(3):
        assert np.array_equal(load_image(tmp_path, "e", i), images[i])


def test_write_refuses_overwrite_and_bad_inputs(tmp_path):
    ids = make_episodes(tmp_path, n_episodes=1, steps=3)
    with pytest.raises(FileExistsError):
        make_episodes(tmp_path, n_episodes=1, steps=3)
    arrays = {k: np.zeros(3) for k in ("t", "speed", "command", "steer", "throttle", "brake")}
    meta = {"scenario_name": "s", "seed": 0, "dt": 0.1, "image_height": 4, "image_width": 4}
    good = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        write_episode(tmp_path, "a", {k: v for k, v in meta.items() if k != "dt"}, arrays, good)
    with pytest.raises(ValueError):
        write_episode(tmp_path, "b", meta, {k: v for k, v in arrays.items() if k != "steer"}, good)
    with pytest.raises(ValueError):
        write_episode(tmp_path, "c", meta, arrays, good[:2])
    with pytest.raises(ValueError):
        write_episode(tmp_path, "d", meta, arrays, good.astype(np.float32))
    with pytest.raises(ValueError):
        write_episode(tmp_path, "e", {**meta, "image_height": 5}, arrays, good)
    assert ids == list_episodes(tmp_path)


def test_validation_errors(tmp_path):
    ids = make_episodes(tmp_path, n_episodes=5, steps=4)
    (tmp_path / ids[0] / "meta.json").unlink()
    with pytest.raises(ValueError, match="meta.json"):
        validate_episode(tmp_path, ids[0])
    (tmp_path / ids[1] / "steps.npz").unlink()
    with pytest.raises(ValueError, match="steps.npz"):
        validate_episode(tmp_path, ids[1])
    (tmp_path / ids[2] / "images" / "000002.png").unlink()
    with pytest.raises(ValueError, match="image files"):
        validate_episode(tmp_path, ids[2])
    z = dict(np.load(tmp_path / ids[3] / "steps.npz"))
    z["steer"] = z["steer"][:-1]
    np.savez(tmp_path / ids[3] / "steps.npz", **z)
    with pytest.raises(ValueError, match="length mismatch"):
        validate_episode(tmp_path, ids[3])
    meta = json.loads((tmp_path / ids[4] / "meta.json").read_text(encoding="utf-8"))
    meta["image_width"] = 99
    (tmp_path / ids[4] / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="size"):
        validate_episode(tmp_path, ids[4])


def test_dataset_shapes(tmp_path):
    ids = make_episodes(tmp_path, n_episodes=3, steps=6)
    ds = EpisodeDataset(tmp_path, ids)
    assert len(ds) == 18
    item = ds[7]
    assert item["image"].shape == (3, 32, 64) and item["image"].dtype == torch.float32
    assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    assert item["speed"].dtype == torch.float32 and item["speed"].ndim == 0
    assert item["command"].dtype == torch.int64 and int(item["command"]) in range(4)
    assert item["target"].shape == (3,) and item["target"].dtype == torch.float32
    with pytest.raises(ValueError):
        EpisodeDataset(tmp_path, [])


def test_dataset_target_matches_store(tmp_path):
    ids = make_episodes(tmp_path, n_episodes=1, steps=5)
    _, arrays = read_episode(tmp_path, ids[0])
    ds = EpisodeDataset(tmp_path, ids)
    for i in range(5):
        assert float(ds[i]["target"][0]) == pytest.approx(float(arrays["steer"][i]))
        assert int(ds[i]["command"]) == int(arrays["command"][i])


def test_folds_properties():
    ids = [f"e{i}" for i in range(11)]
    folds = make_folds(ids, 4, seed=1)
    assert len(folds) == 4
    val_all = [x for _, v in folds for x in v]
    assert sorted(val_all) == sorted(ids)
    sizes = [len(v) for _, v in folds]
    assert max(sizes) - min(sizes) <= 1
    for train_ids, val_ids in folds:
        assert not set(train_ids) & set(val_ids)
        assert sorted(train_ids + val_ids) == sorted(ids)
    assert make_folds(ids, 4, seed=1) == folds
    assert make_folds(ids, 4, seed=2) != folds


def test_folds_errors():
    with pytest.raises(ValueError):
        make_folds(["a", "b"], 1)
    with pytest.raises(ValueError):
        make_folds(["a", "b"], 3)
    with pytest.raises(ValueError):
        make_folds(["a", "a", "b"], 2)


def test_collect_extra_samples_add_rows(tmp_path):
    from carlautopilot.core import Action
    from carlautopilot.data.collect import collect_episode
    from carlautopilot.data.episode_store import read_episode
    from carlautopilot.policies import PathFollowPolicy
    from carlautopilot.scenarios_lib import straight
    from carlautopilot.sim import FakeEnv

    extra = lambda obs, a: [(obs.image, Action(0.5, a.throttle, a.brake))]
    info = collect_episode(FakeEnv(render_images=True, privileged_obs=True), PathFollowPolicy(), straight(), tmp_path, "e", max_steps=10, extra_samples=extra)
    _, arrays = read_episode(tmp_path, "e")
    assert info["steps"] == 20
    assert list(arrays["steer"][1::2]) == [0.5] * 10


def test_collect_stores_expert_target_speed_only_when_available(tmp_path):
    from carlautopilot.core import Action
    from carlautopilot.data.collect import collect_episode
    from carlautopilot.data.episode_store import read_episode
    from carlautopilot.policies import ConstantPolicy, PathFollowPolicy
    from carlautopilot.scenarios_lib import straight
    from carlautopilot.sim import FakeEnv

    extra = lambda obs, a: [(obs.image, a)]
    collect_episode(FakeEnv(render_images=True, privileged_obs=True), PathFollowPolicy(speed_limit_mps=8.0, max_lat_accel=2.0), straight(), tmp_path, "e", max_steps=10, extra_samples=extra)
    _, arrays = read_episode(tmp_path, "e")
    assert len(arrays["target_speed"]) == len(arrays["steer"]) == 20
    assert float(arrays["target_speed"].max()) == 8.0
    collect_episode(FakeEnv(render_images=True), ConstantPolicy(Action(0.0, 0.3, 0.0)), straight(), tmp_path, "c", max_steps=5)
    assert "target_speed" not in read_episode(tmp_path, "c")[1]


def test_image_cache_matches_png_and_png_level_is_lossless(tmp_path):
    import numpy as np

    from carlautopilot.data.episode_dataset import EpisodeDataset
    from carlautopilot.data.episode_store import load_image, write_episode
    from tests.helpers import make_episodes

    ids = make_episodes(tmp_path / "data", n_episodes=2, steps=5)
    plain = EpisodeDataset(tmp_path / "data", ids)
    cached = EpisodeDataset(tmp_path / "data", ids, cache_dir=tmp_path / "cache")
    for k in range(len(plain)):
        assert (plain[k]["image"] == cached[k]["image"]).all()
    imgs = np.random.default_rng(0).integers(0, 256, (3, 8, 16, 3), dtype=np.uint8)
    arrays = {k: np.zeros(3) for k in ("t", "speed", "steer", "throttle", "brake")}
    arrays["command"] = np.zeros(3, dtype=np.int64)
    meta = {"scenario_name": "x", "seed": 0, "dt": 0.05, "image_height": 8, "image_width": 16}
    write_episode(tmp_path / "p", "lvl1", meta, arrays, imgs, png_compress=1)
    assert all((load_image(tmp_path / "p", "lvl1", i) == imgs[i]).all() for i in range(3))
