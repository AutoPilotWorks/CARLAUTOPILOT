import numpy as np
import pytest
import torch

from carlautopilot.core import Action, Command, Observation, Policy
from carlautopilot.data.collect import collect_dataset, collect_episode
from carlautopilot.data.episode_store import list_episodes, read_episode, validate_episode
from carlautopilot.models.route_steering import RouteSteeringNet
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.policies.model_policy import ModelPolicy
from carlautopilot.scenarios_lib import curve, random_scenarios, straight
from carlautopilot.sim import FakeEnv
from carlautopilot.train.train_steering import TrainConfig, train


def render_env():
    return FakeEnv(render_images=True, privileged_obs=True)


def lane_center_columns(img):
    ys, xs = np.where((img[..., 0] == 140) & (img[..., 1] == 140))
    return xs[ys > img.shape[0] * 0.7]


def test_render_shape_and_content():
    env = FakeEnv(render_images=True, image_size=(48, 96))
    obs = env.reset(straight())
    assert obs.image.shape == (48, 96, 3) and obs.image.dtype == np.uint8
    assert (obs.image == 255).all(axis=2).sum() > 20
    assert FakeEnv().reset(straight()).image is None


def test_render_lane_shifts_with_pose():
    right = FakeEnv(render_images=True)
    right.reset(straight(500.0, speed_limit_mps=13.9))
    for _ in range(20):
        obs = right.step(Action(0.4, 0.3, 0.0)).obs
    cols = lane_center_columns(obs.image)
    assert len(cols) > 0 and cols.mean() < 64
    left = FakeEnv(render_images=True)
    left.reset(straight(500.0))
    for _ in range(20):
        obs = left.step(Action(-0.4, 0.3, 0.0)).obs
    cols = lane_center_columns(obs.image)
    assert len(cols) > 0 and cols.mean() > 64


def test_collect_episode_store_and_stride(tmp_path):
    info = collect_episode(render_env(), PathFollowPolicy(), straight(60.0), tmp_path, "e0", stride=5)
    assert info["termination"] == "route_complete"
    n = validate_episode(tmp_path, "e0")
    assert n == info["steps"]
    full = collect_episode(render_env(), PathFollowPolicy(), straight(60.0), tmp_path, "e1", stride=1)
    assert full["steps"] > 4 * n
    meta, arrays = read_episode(tmp_path, "e0")
    assert meta["scenario_name"] == "straight" and meta["image_height"] == 64
    assert np.all(np.abs(arrays["steer"]) <= 1.0) and np.all(arrays["throttle"] >= 0)


def test_collect_labels_are_clean_expert_actions_under_noise(tmp_path):
    collect_episode(render_env(), PathFollowPolicy(speed_limit_mps=7.0), curve("left"), tmp_path, "n", steer_noise_std=0.3, noise_seed=1)
    _, arrays = read_episode(tmp_path, "n")
    assert arrays["steer"].std() > 0.0


def test_collect_noise_is_seeded(tmp_path):
    a = collect_episode(render_env(), PathFollowPolicy(speed_limit_mps=7.0), curve("right"), tmp_path, "a", steer_noise_std=0.3, noise_seed=2, stride=3)
    b = collect_episode(render_env(), PathFollowPolicy(speed_limit_mps=7.0), curve("right"), tmp_path, "b", steer_noise_std=0.3, noise_seed=2, stride=3)
    assert a["steps"] == b["steps"] and a["termination"] == b["termination"]
    _, aa = read_episode(tmp_path, "a")
    _, bb = read_episode(tmp_path, "b")
    assert np.array_equal(aa["steer"], bb["steer"])


def test_collect_requires_images(tmp_path):
    with pytest.raises(ValueError, match="images"):
        collect_episode(FakeEnv(privileged_obs=True), PathFollowPolicy(), straight(60.0), tmp_path, "x")


def test_collect_refuses_existing_episode(tmp_path):
    collect_episode(render_env(), PathFollowPolicy(), straight(40.0), tmp_path, "dup", stride=5)
    with pytest.raises(FileExistsError):
        collect_episode(render_env(), PathFollowPolicy(), straight(40.0), tmp_path, "dup", stride=5)


def test_collect_dataset(tmp_path):
    scs = random_scenarios(3, seed=0, families=("straight",))
    infos = collect_dataset(render_env, PathFollowPolicy, scs, tmp_path, stride=8, prefix="d")
    assert [i["episode_id"] for i in infos] == ["d0000", "d0001", "d0002"]
    assert list_episodes(tmp_path) == ["d0000", "d0001", "d0002"]


def test_random_scenarios_valid_deterministic_and_unique():
    a = random_scenarios(9, seed=5)
    assert a == random_scenarios(9, seed=5) and a != random_scenarios(9, seed=6)
    assert len({s.name for s in a}) == 9
    for s in a:
        s.validate()
    assert {s.name.split("_")[0] for s in a} == {"straight", "curve", "lead"}
    with pytest.raises(ValueError):
        random_scenarios(1, families=("nope",))


def test_model_policy_acts_and_respects_ranges():
    torch.manual_seed(0)
    pol = ModelPolicy(RouteSteeringNet(), speed_limit_mps=10.0)
    assert isinstance(pol, Policy)
    img = np.random.default_rng(0).integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    a = pol.act(Observation(t=0.0, speed_mps=3.0, command=Command.LEFT, image=img))
    assert -1.0 <= a.steer <= 1.0 and 0.0 <= a.throttle <= 1.0 and 0.0 <= a.brake <= 1.0
    with pytest.raises(ValueError):
        pol.act(Observation(t=0.0, speed_mps=3.0, command=Command.FOLLOW))


def test_model_policy_loads_checkpoint(tmp_path):
    scs = random_scenarios(4, seed=0, families=("straight",))
    collect_dataset(render_env, PathFollowPolicy, scs, tmp_path / "data", stride=10)
    ids = list_episodes(tmp_path / "data")
    train(TrainConfig(str(tmp_path / "data"), ids[1:], ids[:1], str(tmp_path / "run"), epochs=1, batch_size=8))
    pol = ModelPolicy.from_checkpoint(tmp_path / "run" / "checkpoint.pt")
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    assert -1.0 <= pol.act(Observation(t=0.0, speed_mps=1.0, command=Command.FOLLOW, image=img)).steer <= 1.0


def test_collect_log_names_what_the_car_hit(tmp_path):
    import dataclasses

    from carlautopilot.scenario import LeadVehicleSpec

    class Coast:
        def reset(self):
            pass

        def act(self, obs):
            return Action(0.0, 0.0, 0.0)

    sc = dataclasses.replace(straight(), lead=LeadVehicleSpec(20.0, ((0.0, 0.0),)), meta={"initial_speed_mps": 10.0})
    info = collect_episode(render_env(), Coast(), sc, tmp_path, "e0")
    assert info["termination"] == "collision" and info["collision"][0] == "collision_vehicle"
    assert "collision" not in collect_episode(render_env(), PathFollowPolicy(), straight(60.0), tmp_path, "e1")


def test_recovery_views_add_rows_labelled_to_steer_back(tmp_path):
    from carlautopilot.data.episode_store import read_episode

    class ViewEnv(FakeEnv):
        extra_views = ((1.0, 0.0), (-1.0, 0.0))

        def _observe(self):
            obs = super()._observe()
            obs.extras["view_images"] = [obs.image, obs.image]  # the fake renders no shifted camera; labels are the point
            return obs

    info = collect_episode(ViewEnv(render_images=True, privileged_obs=True), PathFollowPolicy(), straight(40.0), tmp_path, "e0", stride=5)
    _, a = read_episode(tmp_path, "e0")
    assert info["steps"] == 3 * (info["steps"] // 3) and a["view"][:3].tolist() == [0, 1, 2]
    centre, left, right = (a["steer"][a["view"] == k] for k in range(3))
    assert (left[2:] > centre[2:] + 0.01).all() and (right[2:] < centre[2:] - 0.01).all()  # seen from the left: steer right


def test_expert_light_stop_sets_where_it_halts():
    from carlautopilot.cli.carla_run import ACC
    from carlautopilot.policies.pathfollow import stop_accel

    # 6 m before a red line at 3 m/s: aiming for 2 m before it brakes less than aiming for the default ~7 m
    assert stop_accel(ACC, 3.0, 6.0, 2.0 - ACC.idm.s0) > stop_accel(ACC, 3.0, 6.0)
