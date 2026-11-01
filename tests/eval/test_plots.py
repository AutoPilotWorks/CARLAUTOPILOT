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


def test_run_episode_saves_frames_before_an_event(tmp_path):
    from carlautopilot.core import Action
    from carlautopilot.policies import ConstantPolicy

    d = tmp_path / "frames"
    r = run_episode(FakeEnv(render_images=True), ConstantPolicy(Action(0.0, 0.6, 0.0)), red_light(), frames_dir=d)
    kinds = [e.kind for rec in r.records for e in rec.events]
    assert "red_light" in kinds
    shots = sorted(p for p in d.iterdir() if p.name.endswith("_red_light"))
    assert shots and 0 < len(list(shots[0].glob("*_main.png"))) <= 20


def test_live_view_frame_places_every_camera_and_the_decision():
    import numpy as np

    from carlautopilot.core import Action, Command, Observation
    from carlautopilot.eval.live_view import compose

    cams = {f: np.full((160, 320, 3), v, np.uint8) for f, v in (("left", 10), ("right", 20), ("rear", 30))}
    obs = Observation(t=0.0, speed_mps=5.0, command=Command.LEFT, image=np.full((160, 320, 3), 40, np.uint8),
                      extras={"light_image": np.full((172, 344, 3), 50, np.uint8), "surround_images": cams})
    img = np.asarray(compose(obs, Action(-0.5, 0.2, 0.0), 6.0, (0.9, 12.0, 40.0, 0.1, 40.0, 15.0, -1.0)))
    assert img.shape == (320, 960, 3)
    # sample each tile away from its name label: LEFT FRONT RIGHT / LIGHT REAR
    assert [int(img[y, x, 0]) for y, x in ((100, 160), (100, 480), (100, 800), (260, 160), (260, 480))] == [10, 40, 20, 50, 30]
    plain = np.asarray(compose(Observation(t=0.0, speed_mps=0.0, command=Command.FOLLOW, image=obs.image), Action()))
    assert plain.shape == (320, 960, 3) and int(plain[100, 160, 0]) == 0  # missing cameras are blank
