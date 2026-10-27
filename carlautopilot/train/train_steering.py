from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from carlautopilot.data.episode_dataset import EpisodeDataset
from carlautopilot.data.episode_store import list_episodes
from carlautopilot.data.splits import make_folds
from carlautopilot.models.route_steering import PERCEPT_RANGE_M, RouteSteeringNet


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
    balance_commands: bool = False
    amp: bool = False
    cache_dir: str | None = None
    resume: bool = False
    predict_speed: bool = False
    perceive: bool = False
    percept_from_image: bool = False
    balance_curves: bool = False
    loss_weights: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.5])


CURVE_SLOW_MPS = 6.0


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    return torch.device(name)


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] <= 4:
        return ((pred - target).abs() * w).sum(dim=1).mean()
    # perception columns: red-light BCE, stop distance only when red, lead gap always, relative speed only with a lead
    red, has_lead = target[:, 4], (target[:, 6] < PERCEPT_RANGE_M).float()
    p = pred[:, 4].clamp(1e-4, 1 - 1e-4)
    terms = [
        -(red * torch.log(p) + (1 - red) * torch.log(1 - p)),
        (pred[:, 5] - target[:, 5]).abs() * red,
        (pred[:, 6] - target[:, 6]).abs(),
        (pred[:, 7] - target[:, 7]).abs() * has_lead,
    ]
    return (((pred[:, :4] - target[:, :4]).abs() * w[:4]).sum(dim=1) + sum(w[4 + k] * t for k, t in enumerate(terms))).mean()


def command_weights(commands: np.ndarray) -> np.ndarray:
    """Per-sample weights so that every route command present is drawn equally often."""
    counts = np.bincount(commands)
    return 1.0 / counts[commands]


def curve_weights(commands: np.ndarray, target_speed: np.ndarray, slow_mps: float = CURVE_SLOW_MPS) -> np.ndarray:
    """Per-sample weights so every (command, slow/fast road-geometry speed) cell is drawn equally often: bends where the
    expert slows are rare, and without this the model under-steers and under-brakes there."""
    cell = commands * 2 + (target_speed < slow_mps)
    return 1.0 / np.bincount(cell)[cell]


def _run_epoch(model, loader, device, w, optimizer=None, scaler=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            image, speed, command, target = (batch[k].to(device) for k in ("image", "speed", "command", "target"))
            with torch.autocast(device.type, enabled=scaler is not None):
                loss = _weighted_l1(model(image, speed, command).float(), target, w)
            if train:
                optimizer.zero_grad()
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            total += float(loss.item()) * len(target)
            count += len(target)
    return total / max(count, 1)


def train(cfg: TrainConfig) -> dict:
    device = _resolve_device(cfg.device)
    out = Path(cfg.out_dir)
    last = out / "last.pt"
    if out.exists() and any(out.iterdir()) and not (cfg.resume and last.exists()):
        raise FileExistsError(f"refusing to write into non-empty {out} (use resume to continue from {last.name})")
    if len(cfg.loss_weights) != (8 if cfg.perceive else 4 if cfg.predict_speed else 3):
        raise ValueError("loss_weights needs 3 values (steer throttle brake), 4 with predict_speed (+ target speed), "
                         "8 with perceive (+ red light, stop distance, lead gap, lead relative speed)")
    if set(cfg.train_ids) & set(cfg.val_ids):
        raise ValueError("train and validation episodes overlap")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_set = EpisodeDataset(cfg.data_root, cfg.train_ids, cfg.cache_dir, cfg.predict_speed, cfg.perceive)
    val_set = EpisodeDataset(cfg.data_root, cfg.val_ids, cfg.cache_dir, cfg.predict_speed, cfg.perceive)
    gen = torch.Generator().manual_seed(cfg.seed)
    if cfg.balance_commands or cfg.balance_curves:
        cmds = np.array([int(train_set._arrays[eid]["command"][i]) for eid, i in train_set._index])
        if cfg.balance_curves:
            ts = np.array([float(train_set._arrays[eid]["target_speed"][i]) for eid, i in train_set._index])
            weights = curve_weights(cmds, ts)
        else:
            weights = command_weights(cmds)
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(cmds), generator=gen)
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, generator=gen, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = RouteSteeringNet(backbone=cfg.backbone, predict_speed=cfg.predict_speed, perceive=cfg.perceive,
                             percept_from_image=cfg.percept_from_image).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda") if cfg.amp and device.type == "cuda" else None
    w = torch.tensor(cfg.loss_weights, dtype=torch.float32, device=device)

    history = []
    if cfg.resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        print(json.dumps({"resumed_after_epoch": len(history)}), flush=True)
    for epoch in range(len(history) + 1, cfg.epochs + 1):
        tr = _run_epoch(model, train_loader, device, w, optimizer, scaler)
        va = _run_epoch(model, val_loader, device, w, scaler=scaler)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(json.dumps(history[-1]), flush=True)
        tmp = out / "last.tmp.pt"
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None, "history": history}, tmp)
        tmp.replace(last)

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
    p.add_argument("--loss-weights", type=float, nargs="+", default=None, help="steer throttle brake [target_speed]")
    p.add_argument("--predict-speed", action="store_true", help="also learn the expert's road-geometry target speed (m/s)")
    p.add_argument("--balance-curves", action="store_true", help="balance (command, slow bend / fast) cells; needs target_speed labels")
    p.add_argument("--perceive", action="store_true", help="also learn red light / lead from the image (needs --predict-speed)")
    p.add_argument("--percept-from-image", action="store_true", help="perception outputs from the image alone (no speed input)")
    p.add_argument("--balance-commands", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--cache-dir", help="decoded-image cache folder (built on first use; pixels identical to the PNGs)")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt (saved after every epoch)")
    a = p.parse_args(argv)
    folds = make_folds(list_episodes(a.data_root), a.k, a.seed)
    if not 0 <= a.fold < len(folds):
        p.error("--fold is out of range")
    train_ids, val_ids = folds[a.fold]
    weights = a.loss_weights or ([1.0, 0.5, 0.5, 0.5, 1.0, 0.05, 0.05, 0.1] if a.perceive else [1.0, 0.5, 0.5, 0.5] if a.predict_speed else [1.0, 0.5, 0.5])
    result = train(TrainConfig(a.data_root, train_ids, val_ids, a.out_dir, a.epochs, a.batch_size, a.lr, a.device, a.seed, a.num_workers, loss_weights=weights, backbone=a.backbone, balance_commands=a.balance_commands, amp=a.amp, cache_dir=a.cache_dir, resume=a.resume, predict_speed=a.predict_speed, perceive=a.perceive, percept_from_image=a.percept_from_image, balance_curves=a.balance_curves))
    print(json.dumps(result["history"][-1]))


if __name__ == "__main__":
    main()
