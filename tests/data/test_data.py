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
    assert item["image"].shape == (3, 32, 64) and item["image"].dtype == torch.uint8  # the model scales it to [0, 1]
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


def test_dagger_driver_steers_expert_labels_and_takes_over(tmp_path):
    from carlautopilot.core import Action
    from carlautopilot.data.collect import collect_episode
    from carlautopilot.data.episode_store import read_episode
    from carlautopilot.policies import ConstantPolicy, PathFollowPolicy
    from carlautopilot.scenarios_lib import straight
    from carlautopilot.sim import FakeEnv

    # a driver that always steers hard left drifts off; the expert must take over and the labels stay the expert's
    info = collect_episode(FakeEnv(render_images=True, privileged_obs=True), PathFollowPolicy(), straight(), tmp_path, "d",
                           max_steps=400, driver=ConstantPolicy(Action(0.5, 0.0, 0.0)))
    steer = read_episode(tmp_path, "d")[1]["steer"]
    assert info["takeovers"] >= 1 and 0.0 < info["driver_share"] < 1.0
    assert info["termination"] != "off_route"
    assert (steer < 0.5).all() and (steer < 0).any()  # expert steers back right, never copies the driver


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


def test_collect_stores_perception_labels_capped_at_range():
    from carlautopilot.core import LeadInfo, Observation, Command
    from carlautopilot.data.collect import perception_labels

    base = dict(t=0.0, speed_mps=5.0, command=Command.FOLLOW)
    assert perception_labels(Observation(**base)) is None
    far = Observation(**base, lead=LeadInfo(80.0, -1.0), extras={"lookahead_points": [(1.0, 0.0)], "next_light_red": True, "next_light_distance_m": 70.0})
    assert perception_labels(far) == (0.0, 40.0, 40.0, 0.0)
    near = Observation(**base, lead=LeadInfo(12.0, -1.5), extras={"lookahead_points": [(1.0, 0.0)], "next_light_red": True, "next_light_distance_m": 20.0})
    assert perception_labels(near) == (1.0, 20.0, 12.0, -1.5)


def test_stop_needed_label_takes_the_nearest_stop_sign_yield_or_red_light():
    from carlautopilot.core import Command, Observation
    from carlautopilot.data.collect import perception_labels

    base = dict(t=0.0, speed_mps=5.0, command=Command.FOLLOW)
    ex = {"lookahead_points": [(1.0, 0.0)], "next_light_red": False, "next_light_distance_m": 10.0}
    assert perception_labels(Observation(**base, extras=ex))[:2] == (0.0, 40.0)
    assert perception_labels(Observation(**base, extras={**ex, "yield_distance_m": 15.0, "next_stop_sign_distance_m": 25.0}))[:2] == (1.0, 15.0)
    assert perception_labels(Observation(**base, extras={**ex, "next_light_red": True, "yield_distance_m": 15.0}))[:2] == (1.0, 10.0)


def test_stop_labels_split_by_cause_and_keep_served_signs():
    from carlautopilot.core import Command, Observation
    from carlautopilot.data.collect import stop_labels

    base = {"lookahead_points": [(5.0, 0.0)]}
    obs = lambda **x: Observation(t=0.0, speed_mps=0.0, command=Command.FOLLOW, extras={**base, **x})
    assert stop_labels(Observation(t=0.0, speed_mps=0.0, command=Command.FOLLOW)) is None
    assert stop_labels(obs()) == (0.0, 40.0, 40.0, 0.0, 40.0)
    assert stop_labels(obs(next_light_distance_m=12.0, next_light_red=False)) == (0.0, 40.0, 40.0, 0.0, 40.0)
    assert stop_labels(obs(next_light_distance_m=12.0, next_light_red=True)) == (1.0, 12.0, 40.0, 0.0, 40.0)
    # a served sign (no next_stop_sign_distance_m) is still seen
    assert stop_labels(obs(stop_sign_distance_m=7.0, yield_distance_m=9.0)) == (0.0, 40.0, 7.0, 1.0, 9.0)
    assert stop_labels(obs(stop_sign_distance_m=60.0)) == (0.0, 40.0, 40.0, 0.0, 40.0)


def test_light_camera_images_round_trip_through_store_cache_and_dataset(tmp_path):
    import numpy as np
    import pytest

    from carlautopilot.data.episode_dataset import EpisodeDataset
    from carlautopilot.data.episode_store import load_image, validate_episode, write_episode
    from tests.helpers import make_episodes

    ids = make_episodes(tmp_path / "d", n_episodes=1, steps=6, light=True)
    assert validate_episode(tmp_path / "d", ids[0]) == 6
    plain = EpisodeDataset(tmp_path / "d", ids, light=True)
    cached = EpisodeDataset(tmp_path / "d", ids, cache_dir=tmp_path / "c", light=True)
    for i in range(6):
        want = load_image(tmp_path / "d", ids[0], i // 2, "light")
        assert np.array_equal(plain[i]["light"].permute(1, 2, 0).numpy(), want)
        assert np.array_equal(plain[i]["light"].numpy(), cached[i]["light"].numpy())
    assert "light" not in EpisodeDataset(tmp_path / "d", ids)[0]
    a = {"t": np.zeros(2), "speed": np.zeros(2, np.float32), "command": np.zeros(2, np.int64), "steer": np.zeros(2, np.float32),
         "throttle": np.zeros(2, np.float32), "brake": np.zeros(2, np.float32), "light_idx": np.array([0, 1])}
    meta = {"scenario_name": "x", "seed": 0, "dt": 0.1, "image_height": 4, "image_width": 4}
    img = np.zeros((2, 4, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        write_episode(tmp_path / "e", "bad", meta, a, img, step_images={"light": img[:1]})
    make_episodes(tmp_path / "d2", n_episodes=1, steps=2)
    with pytest.raises(ValueError):
        EpisodeDataset(tmp_path / "d2", ["ep000"], light=True)
    with pytest.raises(ValueError):  # surround cameras without the light images
        write_episode(tmp_path / "e", "no_light", meta, a, img, step_images={"left": img})
    with pytest.raises(ValueError):  # a surround camera short of one image per step
        write_episode(tmp_path / "e", "short_left", meta, a, img, step_images={"light": img, "left": img[:1]})


def test_multicam_collect_stores_surround_per_step(tmp_path):
    import numpy as np
    import pytest

    from carlautopilot.data.collect import collect_episode
    from carlautopilot.data.episode_store import load_image, read_episode, validate_episode
    from carlautopilot.policies import PathFollowPolicy
    from carlautopilot.scenarios_lib import straight
    from carlautopilot.sim import FakeEnv

    class MultiCam(FakeEnv):
        """What CarlaEnv(multicam=True) adds to every observation; pixel value = env step + camera offset."""
        n = 0

        def _cams(self, obs):
            img = lambda v: np.full((8, 16, 3), v % 256, np.uint8)
            obs.extras.update(light_image=img(self.n), surround_images={"left": img(self.n + 1), "right": img(self.n + 2), "rear": img(self.n + 3)})
            self.n += 1
            return obs

        def reset(self, *a, **k):
            return self._cams(super().reset(*a, **k))

        def step(self, *a, **k):
            res = super().step(*a, **k)
            self._cams(res.obs)
            return res

    collect_episode(MultiCam(render_images=True, privileged_obs=True), PathFollowPolicy(), straight(), tmp_path, "m", max_steps=10, stride=2)
    _, arrays = read_episode(tmp_path, "m")
    assert list(arrays["light_idx"]) == [0, 1, 2, 3, 4]
    for j in range(5):  # stride 2: stored step j is env step 2j, and every camera keeps its own image of that step
        for folder, off in (("light", 0), ("left", 1), ("right", 2), ("rear", 3)):
            assert load_image(tmp_path, "m", j, folder)[0, 0, 0] == 2 * j + off
    (tmp_path / "m" / "rear" / "000004.png").unlink()
    with pytest.raises(ValueError, match="rear camera files"):
        validate_episode(tmp_path, "m")
