from __future__ import annotations

import argparse
import json
import math
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
from carlautopilot.sim.fake_env import STOP_SIGN_ZONE_M


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
    split_stops: bool = False
    balance_stops: bool = False
    light_camera: bool = False
    boost_red_approach: float = 1.0
    boost_yield: float = 1.0
    cosine: bool = False
    loss_weights: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.5])


CURVE_SLOW_MPS = 6.0


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    return torch.device(name)


def _bce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p))


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] == 11:
        # split stops: red BCE, its line only when red, sign distance always (range = none), yield BCE, its entry only
        # when yielding, lead gap always, relative speed only with a lead
        red, yld, has_lead = target[:, 4], target[:, 7], (target[:, 9] < PERCEPT_RANGE_M).float()
        terms = [
            _bce(pred[:, 4], red), (pred[:, 5] - target[:, 5]).abs() * red, (pred[:, 6] - target[:, 6]).abs(),
            _bce(pred[:, 7], yld), (pred[:, 8] - target[:, 8]).abs() * yld,
            (pred[:, 9] - target[:, 9]).abs(), (pred[:, 10] - target[:, 10]).abs() * has_lead,
        ]
        return (((pred[:, :4] - target[:, :4]).abs() * w[:4]).sum(dim=1) + sum(w[4 + k] * t for k, t in enumerate(terms))).mean()
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


STOP_SLOW_MPS = 1.0


def stop_weights(speed: np.ndarray, stop_needed: np.ndarray) -> np.ndarray:
    """Per-sample weights so (standing/moving, stop needed or not) cells are drawn equally often: standing frames are
    almost all 'stop needed', so a model can say stop for any standstill-looking scene without reading the light."""
    cell = (speed < STOP_SLOW_MPS) * 2 + stop_needed
    return 1.0 / np.bincount(cell, minlength=4)[cell]


RED_APPROACH_M = 30.0


def red_approach_weights(red: np.ndarray, speed: np.ndarray, red_dist: np.ndarray, boost: float) -> np.ndarray:
    """boost for frames driving up to a red light within RED_APPROACH_M: where v1.10c misjudged the stop line by +5 m in an
    unseen town and braked too late (Town02, 2026-10-28)."""
    return np.where((red > 0.5) & (speed >= STOP_SLOW_MPS) & (red_dist < RED_APPROACH_M), boost, 1.0)


def cosine_lr(lr: float, epoch: int, epochs: int) -> float:
    """Per-epoch cosine decay from lr (epoch 1) to near 0 (last epoch); set from the epoch number, so resume needs no state."""
    return lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / epochs))


EXCLUDE_FILE = "exclude_rows.json"


def excluded_rows(root: str | Path, index: list[tuple[str, int]]) -> np.ndarray:
    """True for rows listed in <root>/exclude_rows.json ({episode: [[first, last], ...]}): rows whose route command was
    wrong at a junction (planner said FOLLOW where the route turns), so the expert's turn contradicts the command."""
    path = Path(root) / EXCLUDE_FILE
    spans = json.loads(path.read_text()) if path.is_file() else {}
    return np.array([any(a <= i <= b for a, b in spans.get(eid, ())) for eid, i in index], dtype=bool)


def _run_epoch(model, loader, device, w, optimizer=None, scaler=None) -> float:
    train = optimizer is not None
    model.train(train)
    # summed on the device in float64 (the same sum the old per-batch loss.item() made in Python): .item() every batch
    # made the main process wait for the GPU each step
    total, count = torch.zeros((), dtype=torch.float64, device=device), 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            image, speed, command, target = (batch[k].to(device, non_blocking=True) for k in ("image", "speed", "command", "target"))
            light = batch["light"].to(device, non_blocking=True) if "light" in batch else None
            with torch.autocast(device.type, enabled=scaler is not None):
                loss = _weighted_l1(model(image, speed, command, light).float(), target, w)
            if train:
                optimizer.zero_grad()
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            total += loss.detach().double() * len(target)
            count += len(target)
    return float(total.item()) / max(count, 1)


def train(cfg: TrainConfig) -> dict:
    device = _resolve_device(cfg.device)
    out = Path(cfg.out_dir)
    last = out / "last.pt"
    if out.exists() and any(out.iterdir()) and not (cfg.resume and last.exists()):
        raise FileExistsError(f"refusing to write into non-empty {out} (use resume to continue from {last.name})")
    if len(cfg.loss_weights) != ((11 if cfg.split_stops else 8) if cfg.perceive else 4 if cfg.predict_speed else 3):
        raise ValueError("loss_weights needs 3 values (steer throttle brake), 4 with predict_speed (+ target speed), "
                         "8 with perceive (+ red light, stop distance, lead gap, lead relative speed), 11 with split_stops "
                         "(+ red, its line, stop sign, yield, its entry, lead gap, lead relative speed)")
    if set(cfg.train_ids) & set(cfg.val_ids):
        raise ValueError("train and validation episodes overlap")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_set = EpisodeDataset(cfg.data_root, cfg.train_ids, cfg.cache_dir, cfg.predict_speed, cfg.perceive, cfg.split_stops, cfg.light_camera)
    val_set = EpisodeDataset(cfg.data_root, cfg.val_ids, cfg.cache_dir, cfg.predict_speed, cfg.perceive, cfg.split_stops, cfg.light_camera)
    gen = torch.Generator().manual_seed(cfg.seed)
    # pinned batches copy to the GPU without blocking; persistent workers skip re-spawning (and re-opening every cache
    # memmap) at each epoch, which is slow on Windows
    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0)
    excluded = excluded_rows(cfg.data_root, train_set._index)
    if cfg.balance_commands or cfg.balance_curves or cfg.balance_stops or cfg.boost_red_approach != 1.0 or cfg.boost_yield != 1.0 or excluded.any():
        cmds = np.array([int(train_set._arrays[eid]["command"][i]) for eid, i in train_set._index])
        col = lambda k: np.array([float(train_set._arrays[eid][k][i]) for eid, i in train_set._index])
        if cfg.balance_curves:
            ts = np.array([float(train_set._arrays[eid]["target_speed"][i]) for eid, i in train_set._index])
            weights = curve_weights(cmds, ts)
        else:
            weights = command_weights(cmds)
        if cfg.balance_stops:
            needed = ((col("red") > 0.5) | (col("yield") > 0.5) | (col("sign_dist") < STOP_SIGN_ZONE_M)).astype(int)
            weights = weights * stop_weights(col("speed"), needed)
        if cfg.boost_red_approach != 1.0:
            weights = weights * red_approach_weights(col("red"), col("speed"), col("red_dist"), cfg.boost_red_approach)
        if cfg.boost_yield != 1.0:
            # yield rows are ~1% and balance_stops buckets them with red/stop sign, so v1.11 never predicted yield
            weights = weights * np.where(col("yield") > 0.5, cfg.boost_yield, 1.0)
        weights = weights * ~excluded
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(cmds), generator=gen)
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, generator=gen, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **loader_kw)

    model = RouteSteeringNet(backbone=cfg.backbone, predict_speed=cfg.predict_speed, perceive=cfg.perceive,
                             percept_from_image=cfg.percept_from_image, split_stops=cfg.split_stops,
                             light_camera=cfg.light_camera).to(device)
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
        if cfg.cosine:
            # constant lr left v1.11/v1.12 still swinging at the last epoch (Town02 corner steer +0.17 at epoch 7, +0.05 at 8)
            for g in optimizer.param_groups:
                g["lr"] = cosine_lr(cfg.lr, epoch, cfg.epochs)
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
    p.add_argument("--split-stops", action="store_true", help="perception by cause: red light, stop sign, yield (needs --percept-from-image)")
    p.add_argument("--balance-stops", action="store_true", help="balance (standing/moving, stop needed) cells; needs --split-stops data")
    p.add_argument("--light-camera", action="store_true", help="feed the traffic-light camera images to the perception head")
    p.add_argument("--boost-red-approach", type=float, default=1.0, help="draw frames driving up to a red light (< 30 m) this many times more often; needs --split-stops data")
    p.add_argument("--boost-yield", type=float, default=1.0, help="draw yield frames this many times more often; needs --split-stops data")
    p.add_argument("--cosine", action="store_true", help="cosine learning-rate decay over the epochs (per epoch)")
    p.add_argument("--balance-commands", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--cache-dir", help="decoded-image cache folder (built on first use; pixels identical to the PNGs)")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt (saved after every epoch)")
    a = p.parse_args(argv)
    folds = make_folds(list_episodes(a.data_root), a.k, a.seed)
    if not 0 <= a.fold < len(folds):
        p.error("--fold is out of range")
    train_ids, val_ids = folds[a.fold]
    split = [1.0, 0.5, 0.5, 0.5, 1.0, 0.05, 0.05, 1.0, 0.05, 0.05, 0.1]
    weights = a.loss_weights or ((split if a.split_stops else [1.0, 0.5, 0.5, 0.5, 1.0, 0.05, 0.05, 0.1]) if a.perceive else [1.0, 0.5, 0.5, 0.5] if a.predict_speed else [1.0, 0.5, 0.5])
    result = train(TrainConfig(a.data_root, train_ids, val_ids, a.out_dir, a.epochs, a.batch_size, a.lr, a.device, a.seed, a.num_workers, loss_weights=weights, backbone=a.backbone, balance_commands=a.balance_commands, amp=a.amp, cache_dir=a.cache_dir, resume=a.resume, predict_speed=a.predict_speed, perceive=a.perceive, percept_from_image=a.percept_from_image, balance_curves=a.balance_curves, split_stops=a.split_stops, balance_stops=a.balance_stops, light_camera=a.light_camera,
                                 boost_red_approach=a.boost_red_approach, boost_yield=a.boost_yield, cosine=a.cosine))
    print(json.dumps(result["history"][-1]))


if __name__ == "__main__":
    main()
