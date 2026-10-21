from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from carlautopilot.data.episode_dataset import EpisodeDataset
from carlautopilot.data.episode_store import list_episodes
from carlautopilot.data.splits import make_folds
from carlautopilot.models.route_steering import RouteSteeringNet


@dataclass
class TrainConfig:
    data_root: str
    train_ids: list[str]
    val_ids: list[str]
    out_dir: str
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    num_workers: int = 0
    backbone: str = "small"
    loss_weights: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.5])


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    return torch.device(name)


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs() * w).sum(dim=1).mean()


def _run_epoch(model, loader, device, w, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            image, speed, command, target = (batch[k].to(device) for k in ("image", "speed", "command", "target"))
            loss = _weighted_l1(model(image, speed, command), target, w)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * len(target)
            count += len(target)
    return total / max(count, 1)


def train(cfg: TrainConfig) -> dict:
    device = _resolve_device(cfg.device)
    out = Path(cfg.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")
    if set(cfg.train_ids) & set(cfg.val_ids):
        raise ValueError("train and validation episodes overlap")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_set = EpisodeDataset(cfg.data_root, cfg.train_ids)
    val_set = EpisodeDataset(cfg.data_root, cfg.val_ids)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, generator=gen, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = RouteSteeringNet(backbone=cfg.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    w = torch.tensor(cfg.loss_weights, dtype=torch.float32, device=device)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        tr = _run_epoch(model, train_loader, device, w, optimizer)
        va = _run_epoch(model, val_loader, device, w)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(json.dumps(history[-1]), flush=True)

    metrics = {"history": history, "train_samples": len(train_set), "val_samples": len(val_set), "device": cfg.device}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "cfg": asdict(cfg)}, out / "checkpoint.pt")
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train the route-conditioned steering network")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--backbone", default="small", choices=["small", "resnet18"])
    p.add_argument("--loss-weights", type=float, nargs=3, default=[1.0, 0.5, 0.5], help="steer throttle brake")
    a = p.parse_args(argv)
    folds = make_folds(list_episodes(a.data_root), a.k, a.seed)
    if not 0 <= a.fold < len(folds):
        p.error("--fold is out of range")
    train_ids, val_ids = folds[a.fold]
    result = train(TrainConfig(a.data_root, train_ids, val_ids, a.out_dir, a.epochs, a.batch_size, a.lr, a.device, a.seed, a.num_workers, loss_weights=a.loss_weights, backbone=a.backbone))
    print(json.dumps(result["history"][-1]))


if __name__ == "__main__":
    main()
