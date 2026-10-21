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
