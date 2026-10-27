from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from carlautopilot.control import ACCController, SafetyConfig, apply_safety
from carlautopilot.core import Action, Command, LeadInfo, Observation
from carlautopilot.models.route_steering import PERCEPT_RANGE_M, RouteSteeringNet
from carlautopilot.policies.pathfollow import stop_accel


@dataclass
class ModelPolicy:
    """Steering from the network; speed from IDM/ACC unless use_model_pedals is set. Safety on top.

    A predict_speed model sets the ACC speed limit itself (learned from the expert's road-geometry speed);
    lead vehicles and red lights stay with the ACC rules, unless the model also perceives them (perceive): then the
    simulator's lead is replaced by the predicted one and a predicted red light stops the car at its predicted stop line. Otherwise turn_speed_mps is a rule, not learned:
    while the route command is LEFT/RIGHT/STRAIGHT (a junction), the ACC limit is capped to that speed.
    """

    model: RouteSteeringNet
    speed_limit_mps: float = 13.9
    use_model_pedals: bool = False
    turn_speed_mps: float | None = None
    sim_lead: bool = False  # ablation: keep the simulator's lead even when the model perceives one
    acc: ACCController = field(default_factory=ACCController)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    device: str = "cpu"
    name: str = "model"

    def __post_init__(self) -> None:
        self.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kw) -> "ModelPolicy":
        ckpt = torch.load(path, map_location=kw.get("device", "cpu"), weights_only=False)
        cfg = ckpt.get("cfg", {})
        model = RouteSteeringNet(backbone=cfg.get("backbone", "small"), pretrained=False, predict_speed=cfg.get("predict_speed", False),
                                 perceive=cfg.get("perceive", False), percept_from_image=cfg.get("percept_from_image", False))
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
        red = False
        if self.model.perceive:
            red = out[4] > 0.5
            lead = LeadInfo(float(out[6]), float(out[7])) if out[6] < PERCEPT_RANGE_M - 2.0 else None
            if not self.sim_lead:
                obs = replace(obs, lead=lead)
        if self.use_model_pedals:
            throttle, brake = float(out[1]), float(out[2])
        else:
            limit = self.speed_limit_mps
            if self.model.predict_speed:
                limit = min(limit, float(out[3]))
            elif self.turn_speed_mps is not None and obs.command != Command.FOLLOW:
                limit = min(limit, self.turn_speed_mps)
            accel = self.acc.accel(obs, limit)
            if red:
                accel = min(accel, stop_accel(self.acc, obs.speed_mps, float(out[5])))
            throttle, brake = self.acc.pedals(accel)
        return apply_safety(Action(steer, throttle, brake), obs, self.safety)
