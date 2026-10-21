from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from carlautopilot.control import ACCController, SafetyConfig, apply_safety
from carlautopilot.core import Action, Command, Observation
from carlautopilot.models.route_steering import RouteSteeringNet


@dataclass
class ModelPolicy:
    """Steering from the network; speed from IDM/ACC unless use_model_pedals is set. Safety on top.

    turn_speed_mps is a rule, not learned: while the route command is LEFT/RIGHT/STRAIGHT (a junction),
    the ACC speed limit is capped so the car enters turns at speeds the expert data covers.
    """

    model: RouteSteeringNet
    speed_limit_mps: float = 13.9
    use_model_pedals: bool = False
    turn_speed_mps: float | None = None
    acc: ACCController = field(default_factory=ACCController)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    device: str = "cpu"
    name: str = "model"

    def __post_init__(self) -> None:
        self.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kw) -> "ModelPolicy":
        ckpt = torch.load(path, map_location=kw.get("device", "cpu"), weights_only=False)
        model = RouteSteeringNet(backbone=ckpt.get("cfg", {}).get("backbone", "small"), pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, **kw)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: Observation) -> Action:
        if obs.image is None:
            raise ValueError("ModelPolicy needs image observations")
        image = torch.from_numpy(np.array(obs.image, dtype=np.uint8)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        speed = torch.tensor([obs.speed_mps], dtype=torch.float32, device=self.device)
        command = torch.tensor([int(obs.command)], dtype=torch.int64, device=self.device)
        out = self.model(image, speed, command)[0].cpu().numpy()
        steer = float(out[0])
        if self.use_model_pedals:
            throttle, brake = float(out[1]), float(out[2])
        else:
            limit = self.speed_limit_mps
            if self.turn_speed_mps is not None and obs.command != Command.FOLLOW:
                limit = min(limit, self.turn_speed_mps)
            throttle, brake = self.acc.act(obs, limit)
        return apply_safety(Action(steer, throttle, brake), obs, self.safety)
