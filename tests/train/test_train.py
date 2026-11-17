import json

import numpy as np
import pytest
import torch

from carlautopilot.models.route_steering import RouteSteeringNet
from carlautopilot.train.train_steering import TrainConfig, main, train
from tests.helpers import make_episodes


def test_waypoint_model_learns_the_path_per_command_and_the_policy_follows_it(tmp_path):
    from carlautopilot.core import Command, Observation
    from carlautopilot.policies.model_policy import ModelPolicy

    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=40, waypoints=True)
    res = train(cfg(tmp_path, ids, epochs=12, waypoints=True))
    assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"] * 0.5
    pol = ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt")
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    steers = {}
    for c in (Command.LEFT, Command.RIGHT):
        pol.reset()
        steers[c] = pol.act(Observation(t=0.0, speed_mps=5.0, command=c, image=img, extras={"route_target": (30.0, 0.0)})).steer
        assert len(pol.last_waypoints) == 8
    assert steers[Command.LEFT] < -0.05 and steers[Command.RIGHT] > 0.05  # positive steer is right
    with pytest.raises(ValueError, match="route_target"):
        pol.act(Observation(t=0.0, speed_mps=5.0, command=Command.LEFT, image=img))


def cfg(tmp_path, ids, epochs=2, **kw):
    return TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=epochs, batch_size=8, lr=3e-3, **kw)


def test_tiny_run_writes_outputs(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=8)
    res = train(cfg(tmp_path, ids))
    assert len(res["history"]) == 2 and res["train_samples"] == 24 and res["val_samples"] == 8
    assert all(torch.isfinite(torch.tensor(h["train_loss"])) for h in res["history"])
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["history"] == res["history"]
    ckpt = torch.load(tmp_path / "out" / "checkpoint.pt", weights_only=False)
    net = RouteSteeringNet()
    net.load_state_dict(ckpt["model_state"])
    assert ckpt["cfg"]["epochs"] == 2


def test_overfit_loss_drops(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=20)
    c = cfg(tmp_path, ids, epochs=25, seed=1)
    res = train(c)
    first, last = res["history"][0]["train_loss"], res["history"][-1]["train_loss"]
    assert last < 0.5 * first


def test_deterministic_given_seed(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=8)
    a = train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "o1"), epochs=2, batch_size=8, seed=5))
    b = train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "o2"), epochs=2, batch_size=8, seed=5))
    assert a["history"] == b["history"]


def test_refuses_nonempty_out_dir_and_overlap(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        train(cfg(tmp_path, ids))
    with pytest.raises(ValueError):
        train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[2:], str(tmp_path / "other"), epochs=1))


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA exists on this machine")
def test_cuda_unavailable_error(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4)
    with pytest.raises(RuntimeError, match="CUDA"):
        train(cfg(tmp_path, ids, device="cuda"))


def test_bad_device_name(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4)
    with pytest.raises(ValueError):
        train(cfg(tmp_path, ids, device="tpu"))


def test_cli_runs_one_fold(tmp_path, capsys):
    make_episodes(tmp_path / "data", n_episodes=4, steps=6)
    main(["--data-root", str(tmp_path / "data"), "--out-dir", str(tmp_path / "cli"), "--k", "2", "--fold", "1", "--epochs", "1", "--batch-size", "8"])
    assert "train_loss" in capsys.readouterr().out
    assert (tmp_path / "cli" / "checkpoint.pt").is_file()


def test_cli_passes_loss_weights_and_backbone(tmp_path):
    from carlautopilot.train.train_steering import main

    make_episodes(tmp_path / "data", n_episodes=4, steps=4)
    main(["--data-root", str(tmp_path / "data"), "--out-dir", str(tmp_path / "out"), "--k", "2", "--epochs", "1", "--loss-weights", "10", "0.5", "0.25"])
    cfg = torch.load(tmp_path / "out" / "checkpoint.pt", weights_only=False)["cfg"]
    assert cfg["loss_weights"] == [10.0, 0.5, 0.25]
    assert cfg["backbone"] == "small"


def test_command_weights_equalize_commands():
    import numpy as np

    from carlautopilot.train.train_steering import command_weights

    cmds = np.array([0] * 8 + [1] * 2 + [3] * 1)
    w = command_weights(cmds)
    totals = [w[cmds == c].sum() for c in (0, 1, 3)]
    assert totals == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
def test_balanced_and_amp_training_runs(tmp_path, device):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=6)
    res = train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, device=device, balance_commands=True, amp=True))
    assert res["history"][0]["train_loss"] > 0


def test_cosine_lr_decays_to_near_zero_and_trains(tmp_path):
    from carlautopilot.train.train_steering import cosine_lr

    lrs = [cosine_lr(1.0, e, 8) for e in range(1, 9)]
    assert lrs[0] == 1.0 and all(a > b for a, b in zip(lrs, lrs[1:])) and lrs[-1] < 0.05
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4)
    assert train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=2, cosine=True))["history"]


def test_resume_continues_after_the_last_saved_epoch(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4)
    first = train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1))
    assert (tmp_path / "out" / "last.pt").exists()
    with pytest.raises(FileExistsError):
        train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=2))
    both = train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=2, resume=True))
    assert [h["epoch"] for h in both["history"]] == [1, 2]
    assert both["history"][0] == first["history"][0]


def test_predict_speed_trains_a_fourth_output_and_needs_matching_weights(tmp_path):
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4, target_speed=True)
    with pytest.raises(ValueError):
        train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "bad"), epochs=1, predict_speed=True))
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, loss_weights=[1.0, 0.5, 0.5, 0.5]))
    from carlautopilot.policies.model_policy import ModelPolicy

    assert ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt").model.predict_speed


def test_perception_losses_mask_distances_that_do_not_apply():
    import torch

    from carlautopilot.train.train_steering import _weighted_l1

    # no red, no yield, no lead: their line / entry / relative speed are ignored; the sign distance always counts
    target = torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0, 40.0, 40.0, 0.0, 40.0, 40.0, 0.0]])
    pred = target.clone()
    pred[0, 4], pred[0, 7] = 1e-4, 1e-4
    pred[0, 5], pred[0, 8], pred[0, 10] = 10.0, 5.0, 3.0
    assert _weighted_l1(pred, target, torch.ones(11)).item() < 1e-3
    pred[0, 6] = 30.0
    assert _weighted_l1(pred, target, torch.ones(11)).item() > 9.0


def test_curve_weights_draw_each_command_speed_cell_equally():
    import numpy as np

    from carlautopilot.train.train_steering import curve_weights

    cmds = np.array([0] * 8 + [0, 1, 1, 1])
    ts = np.array([8.0] * 8 + [4.0, 4.0, 8.0, 8.0])
    w = curve_weights(cmds, ts)
    cells = cmds * 2 + (ts < 6.0)
    totals = [w[cells == c].sum() for c in np.unique(cells)]
    assert np.allclose(totals, 1.0)


def test_perceive_ignores_speed_and_trains_eleven_outputs_with_stop_balancing(tmp_path):
    import numpy as np
    import torch

    from carlautopilot.models.route_steering import RouteSteeringNet
    from carlautopilot.policies.model_policy import ModelPolicy
    from carlautopilot.train.train_steering import EXCLUDE_FILE, excluded_rows, red_approach_weights, stop_weights

    with pytest.raises(ValueError):
        RouteSteeringNet(perceive=True)  # needs predict_speed
    net = RouteSteeringNet(predict_speed=True, perceive=True).eval()
    image, command = torch.rand(1, 3, 32, 64), torch.tensor([0])
    stopped, moving = net(image, torch.tensor([0.0]), command), net(image, torch.tensor([8.0]), command)
    assert stopped.shape == (1, 11)
    assert torch.allclose(stopped[:, 4:], moving[:, 4:])  # standing still must not imply 'stop needed'
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=6, target_speed=True, perception=True)
    with pytest.raises(ValueError):
        train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "bad"), epochs=1, predict_speed=True, perceive=True,
                          loss_weights=[1.0] * 8))
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, perceive=True,
                      balance_stops=True, boost_red_approach=3.0, boost_yield=5.0, loss_weights=[1.0] * 11))
    pol = ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt")
    assert pol.model.perceive and pol.model.n_out == 11
    w = stop_weights(np.array([0.0, 0.0, 0.0, 5.0]), np.array([1, 1, 0, 0]))
    assert np.allclose(w, [0.5, 0.5, 1.0, 1.0])
    w = red_approach_weights(np.array([1, 1, 1, 0]), np.array([5.0, 0.0, 5.0, 5.0]), np.array([20.0, 20.0, 35.0, 20.0]), 3.0)
    assert np.allclose(w, [3.0, 1.0, 1.0, 1.0])  # only moving, red, within 30 m
    (tmp_path / "data" / EXCLUDE_FILE).write_text('{"%s": [[1, 2]]}' % ids[0])
    assert excluded_rows(tmp_path / "data", [(ids[0], 0), (ids[0], 1), (ids[0], 2), (ids[0], 3), (ids[1], 1)]).tolist() == [False, True, True, False, False]
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out2"), epochs=1, predict_speed=True, perceive=True,
                      loss_weights=[1.0] * 11))  # exclusion alone turns the sampler on


def test_light_camera_model_trains_and_needs_the_light_image(tmp_path):
    import torch

    from carlautopilot.models.route_steering import RouteSteeringNet
    from carlautopilot.policies.model_policy import ModelPolicy

    with pytest.raises(ValueError):
        RouteSteeringNet(light_camera=True)
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=6, target_speed=True, perception=True, light=True)
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, perceive=True,
                      light_camera=True, loss_weights=[1.0] * 11))
    m = ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt").model
    assert m.light_camera
    x, v, c = torch.zeros(1, 3, 32, 64), torch.zeros(1), torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError):
        m(x, v, c)
    assert m(x, v, c, torch.zeros(1, 3, 16, 32)).shape == (1, 11)
    # P(red) (column 4) comes from the light image alone: v1.10 read it off the main view and ignored the light camera
    m.eval()
    light = torch.rand(2, 3, 16, 32)
    with torch.no_grad():
        a, b = m(torch.zeros(2, 3, 32, 64), v.expand(2), c.expand(2), light), m(torch.rand(2, 3, 32, 64), v.expand(2), c.expand(2), light)
        assert torch.allclose(a[:, 4], b[:, 4]) and not torch.allclose(a[:, 5:], b[:, 5:])
        assert not torch.allclose(a[:, 4], m(torch.zeros(2, 3, 32, 64), v.expand(2), c.expand(2), light.flip(0))[:, 4])


def test_surround_model_trains_and_reads_left_right_rear_in_order(tmp_path):
    import numpy as np
    import torch

    from carlautopilot.core import Observation
    from carlautopilot.data.episode_dataset import EpisodeDataset
    from carlautopilot.data.episode_store import load_image
    from carlautopilot.policies.model_policy import ModelPolicy

    root = tmp_path / "data"
    ids = make_episodes(root, n_episodes=4, steps=6, target_speed=True, perception=True, light=True, surround=True)
    item, cached = EpisodeDataset(root, ids, light=True, surround=True)[3], EpisodeDataset(root, ids, tmp_path / "c", light=True, surround=True)[3]
    assert item["surround"].shape == (3, 3, 32, 64) and item["surround"].dtype == torch.uint8
    for k, f in enumerate(("left", "right", "rear")):  # row 3 is time step 1
        assert np.array_equal(item["surround"][k].permute(1, 2, 0).numpy(), load_image(root, ids[0], 1, f))
    assert torch.equal(item["surround"], cached["surround"])
    plain = make_episodes(tmp_path / "plain", n_episodes=1, steps=2, light=True)
    with pytest.raises(ValueError):  # light-only data has no surround folders
        EpisodeDataset(tmp_path / "plain", plain, surround=True)

    train(TrainConfig(str(root), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, perceive=True,
                      light_camera=True, surround=True, loss_weights=[1.0] * 11, cache_dir=str(tmp_path / "c")))
    policy = ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt")
    m = policy.model
    assert m.surround
    x, v, c, light = torch.zeros(2, 3, 32, 64), torch.zeros(2), torch.zeros(2, dtype=torch.int64), torch.rand(2, 3, 16, 32)
    with pytest.raises(ValueError):
        m(x, v, c, light)
    s = torch.randint(0, 256, (2, 3, 3, 32, 64), dtype=torch.uint8)
    with torch.no_grad():
        a, b = m(x, v, c, light, s), m(x, v, c, light, s[:, [1, 0, 2]])  # left and right swapped
    assert a.shape == (2, 11)
    assert torch.allclose(a[:, 4], b[:, 4])  # P(red) stays with the light image alone
    assert not torch.allclose(a[:, :4], b[:, :4])  # the camera order matters to the driving heads

    views = {f: np.zeros((32, 64, 3), np.uint8) for f in ("left", "right", "rear")}
    o = Observation(t=0.0, speed_mps=0.0, command=0, image=np.zeros((32, 64, 3), np.uint8), extras={"light_image": np.zeros((16, 32, 3), np.uint8)})
    with pytest.raises(ValueError):
        policy.act(o)
    o.extras["surround_images"] = views
    policy.act(o)
