from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from carlautopilot.data.collect import collect_dataset
from carlautopilot.data.episode_store import list_episodes
from carlautopilot.policies import PathFollowPolicy
from carlautopilot.policies.model_policy import ModelPolicy
from carlautopilot.runner import run_suite
from carlautopilot.scenarios_lib import random_scenarios
from carlautopilot.sim import FakeEnv
from carlautopilot.train.train_steering import TrainConfig, train


def run_dryrun(
    work_dir: str | Path,
    episodes: int = 36,
    heldout: int = 12,
    stride: int = 4,
    epochs: int = 12,
    steer_noise_std: float = 0.25,
    speed_limit_mps: float = 7.0,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Collect with the expert in FakeEnv, train the steering net, evaluate it closed-loop on held-out scenarios."""
    work = Path(work_dir)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {work}")
    work.mkdir(parents=True, exist_ok=True)
    families = ("straight", "curve", "curve")
    t0 = time.time()
    train_scs = random_scenarios(episodes, seed=seed + 1, families=families)
    collect_dataset(
        lambda: FakeEnv(render_images=True, privileged_obs=True),
        lambda: PathFollowPolicy(speed_limit_mps=speed_limit_mps),
        train_scs,
        work / "data",
        steer_noise_std=steer_noise_std,
        seed=seed,
        stride=stride,
    )
    t_collect = time.time() - t0
    ids = list_episodes(work / "data")
    n_val = max(1, len(ids) // 8)
    t1 = time.time()
    res = train(TrainConfig(str(work / "data"), ids[n_val:], ids[:n_val], str(work / "run"), epochs=epochs, batch_size=64, lr=2e-3, seed=seed, device=device))
    t_train = time.time() - t1
    held = random_scenarios(heldout, seed=seed + 99, families=families)
    ckpt = work / "run" / "checkpoint.pt"
    suite = run_suite(
        lambda: FakeEnv(render_images=True),
        lambda: ModelPolicy.from_checkpoint(ckpt, speed_limit_mps=speed_limit_mps, device=device),
        held,
        work / "eval",
        title="Dry run: model policy on held-out FakeEnv scenarios",
    )
    summary = {
        "train_episodes": len(ids) - n_val,
        "val_episodes": n_val,
        "samples": res["train_samples"],
        "final_train_loss": res["history"][-1]["train_loss"],
        "final_val_loss": res["history"][-1]["val_loss"],
        "heldout": suite["aggregate"],
        "seconds": {"collect": round(t_collect, 1), "train": round(t_train, 1)},
    }
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Collect -> train -> closed-loop evaluate, all in the fake environment")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--episodes", type=int, default=36)
    p.add_argument("--heldout", type=int, default=12)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--noise", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args(argv)
    summary = run_dryrun(a.work_dir, a.episodes, a.heldout, a.stride, a.epochs, a.noise, seed=a.seed, device=a.device)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
