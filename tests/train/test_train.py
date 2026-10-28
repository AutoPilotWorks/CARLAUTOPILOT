import json

import pytest
import torch

from carlautopilot.models.route_steering import RouteSteeringNet
from carlautopilot.train.train_steering import TrainConfig, main, train
from tests.helpers import make_episodes


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


def test_perceive_trains_eight_outputs_with_masked_losses(tmp_path):
    import torch

    from carlautopilot.train.train_steering import _weighted_l1

    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4, target_speed=True, perception=True)
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, perceive=True,
                      loss_weights=[1.0, 0.5, 0.5, 0.5, 1.0, 0.05, 0.05, 0.1]))
    from carlautopilot.policies.model_policy import ModelPolicy

    assert ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt").model.perceive
    w = torch.ones(8)
    target = torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0, 50.0, 50.0, 0.0]])
    pred = target.clone()
    pred[0, 4], pred[0, 5], pred[0, 7] = 1e-4, 10.0, 3.0  # stop distance and relative speed are ignored here
    assert _weighted_l1(pred, target, w).item() < 1e-3


def test_curve_weights_draw_each_command_speed_cell_equally():
    import numpy as np

    from carlautopilot.train.train_steering import curve_weights

    cmds = np.array([0] * 8 + [0, 1, 1, 1])
    ts = np.array([8.0] * 8 + [4.0, 4.0, 8.0, 8.0])
    w = curve_weights(cmds, ts)
    cells = cmds * 2 + (ts < 6.0)
    totals = [w[cells == c].sum() for c in np.unique(cells)]
    assert np.allclose(totals, 1.0)


def test_percept_from_image_ignores_speed_and_round_trips(tmp_path):
    import torch

    from carlautopilot.models.route_steering import RouteSteeringNet
    from carlautopilot.policies.model_policy import ModelPolicy

    net = RouteSteeringNet(predict_speed=True, perceive=True, percept_from_image=True).eval()
    image, command = torch.rand(1, 3, 32, 64), torch.tensor([0])
    stopped, moving = net(image, torch.tensor([0.0]), command), net(image, torch.tensor([8.0]), command)
    assert stopped.shape == (1, 8)
    assert torch.allclose(stopped[:, 4:], moving[:, 4:])  # standing still must not imply 'stop needed'
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=4, target_speed=True, perception=True)
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, perceive=True,
                      percept_from_image=True, loss_weights=[1.0, 0.5, 0.5, 0.5, 1.0, 0.05, 0.05, 0.1]))
    assert ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt").model.percept_from_image


def test_split_stops_trains_eleven_outputs_with_stop_balancing(tmp_path):
    import numpy as np

    from carlautopilot.policies.model_policy import ModelPolicy
    from carlautopilot.train.train_steering import red_approach_weights, stop_weights

    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=6, target_speed=True, perception=True, split_stops=True)
    with pytest.raises(ValueError):
        train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "bad"), epochs=1, predict_speed=True, perceive=True,
                          percept_from_image=True, split_stops=True, loss_weights=[1.0] * 8))
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, perceive=True,
                      percept_from_image=True, split_stops=True, balance_stops=True, boost_red_approach=3.0, loss_weights=[1.0] * 11))
    pol = ModelPolicy.from_checkpoint(tmp_path / "out" / "checkpoint.pt")
    assert pol.model.split_stops and pol.model.n_out == 11
    w = stop_weights(np.array([0.0, 0.0, 0.0, 5.0]), np.array([1, 1, 0, 0]))
    assert np.allclose(w, [0.5, 0.5, 1.0, 1.0])
    w = red_approach_weights(np.array([1, 1, 1, 0]), np.array([5.0, 0.0, 5.0, 5.0]), np.array([20.0, 20.0, 35.0, 20.0]), 3.0)
    assert np.allclose(w, [3.0, 1.0, 1.0, 1.0])  # only moving, red, within 30 m


def test_light_camera_model_trains_and_needs_the_light_image(tmp_path):
    import torch

    from carlautopilot.models.route_steering import RouteSteeringNet
    from carlautopilot.policies.model_policy import ModelPolicy

    with pytest.raises(ValueError):
        RouteSteeringNet(light_camera=True)
    ids = make_episodes(tmp_path / "data", n_episodes=4, steps=6, target_speed=True, perception=True, split_stops=True, light=True)
    train(TrainConfig(str(tmp_path / "data"), ids[:3], ids[3:], str(tmp_path / "out"), epochs=1, predict_speed=True, perceive=True,
                      percept_from_image=True, split_stops=True, light_camera=True, loss_weights=[1.0] * 11))
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
