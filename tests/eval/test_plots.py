import numpy as np
import pytest
from PIL import Image

from carlautopilot.cli.plot import main
from carlautopilot.eval.plots import plot_log
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.runner import run_episode
from carlautopilot.scenarios_lib import curve, red_light
from carlautopilot.sim import FakeEnv


def make_log(tmp_path, sc, name):
    p = tmp_path / "logs" / f"{name}.jsonl"
    run_episode(FakeEnv(privileged_obs=True), PathFollowPolicy(), sc, log_path=p)
    return p


def test_plot_log_draws_nonblank_png(tmp_path):
    out = plot_log(make_log(tmp_path, curve("left"), "c"), tmp_path / "c.png")
    img = np.asarray(Image.open(out).convert("RGB"))
    assert img.shape == (760, 900, 3)
    assert (img.sum(axis=2) < 600).sum() > 500


def test_plot_marks_events(tmp_path):
    from carlautopilot.core import Action
    from carlautopilot.policies import ConstantPolicy

    p = tmp_path / "logs" / "r.jsonl"
    run_episode(FakeEnv(), ConstantPolicy(Action(0.0, 0.6, 0.0)), red_light(), log_path=p)
    img = np.asarray(Image.open(plot_log(p, tmp_path / "r.png")).convert("RGB")).astype(int)
    red = (np.abs(img - np.array([210, 30, 30])).sum(axis=2) < 10).sum()
    assert red > 0


def test_cli_and_empty_log(tmp_path):
    make_log(tmp_path, curve("right"), "a")
    make_log(tmp_path, curve("left"), "b")
    paths = main(["--log-dir", str(tmp_path / "logs"), "--out-dir", str(tmp_path / "png")])
    assert len(paths) == 2
    empty = tmp_path / "e.jsonl"
    empty.write_text('{"scenario_name":"x","seed":0,"dt":0.05,"policy_name":"p","schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        plot_log(empty, tmp_path / "e.png")
