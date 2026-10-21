from carlautopilot.data.collect import collect_dataset
from carlautopilot.data.episode_store import list_episodes
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.policies.model_policy import ModelPolicy
from carlautopilot.runner import run_episode
from carlautopilot.scenarios_lib import random_scenarios, straight
from carlautopilot.sim import FakeEnv
from carlautopilot.train.train_steering import TrainConfig, train


def test_collect_train_drive_pipeline_runs_end_to_end(tmp_path):
    """Checks that the stages connect (data format -> training -> inference -> closed-loop run).

    Closed-loop quality depends on dataset size and is reported by the dry-run CLI, not asserted here.
    """
    scs = random_scenarios(10, seed=1, families=("straight", "curve", "curve"))
    collect_dataset(
        lambda: FakeEnv(render_images=True, privileged_obs=True),
        lambda: PathFollowPolicy(speed_limit_mps=7.0),
        scs,
        tmp_path / "data",
        steer_noise_std=0.25,
        stride=6,
    )
    ids = list_episodes(tmp_path / "data")
    res = train(TrainConfig(str(tmp_path / "data"), ids[2:], ids[:2], str(tmp_path / "run"), epochs=8, batch_size=64, lr=2e-3, seed=0))
    hist = res["history"]
    assert hist[-1]["train_loss"] < 0.6 * hist[0]["train_loss"]
    assert hist[-1]["val_loss"] < hist[0]["val_loss"]

    sc = straight(120.0, speed_limit_mps=7.0, name="held_straight")
    pol = ModelPolicy.from_checkpoint(tmp_path / "run" / "checkpoint.pt", speed_limit_mps=7.0)
    r = run_episode(FakeEnv(render_images=True), pol, sc)
    assert len(r.records) >= 30
    assert r.termination in {"route_complete", "off_route", "timeout", "blocked", "collision"}
    assert 0.0 <= r.metrics["driving_score"] <= 1.0
