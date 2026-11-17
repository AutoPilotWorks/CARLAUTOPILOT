from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from carlautopilot.control import ACCController, PurePursuit, SafetyConfig, apply_safety
from carlautopilot.core import Action, LeadInfo, Observation
from carlautopilot.data.episode_store import SURROUND_FOLDERS
from carlautopilot.models.route_steering import N_WAYPOINTS, PERCEPT_RANGE_M, RouteSteeringNet
from carlautopilot.policies.pathfollow import stop_accel
from carlautopilot.sim.fake_env import BLOCKED_SPEED_MPS, STOP_SIGN_HOLD_S, STOP_SIGN_ZONE_M

SIGN_RESET_S = 1.0  # a served stop sign is forgotten once no sign is seen near (within the zone + 5 m) for this long
RED_RELEASE_S = 0.5  # a tracked red light is released after reading clear this long
HOLD_SPEED_MPS, HOLD_LINE_M = 0.5, 12.0  # standstill hold at a red whose line is predicted within HOLD_LINE_M


@dataclass
class ModelPolicy:
    """Steering from the network; pedals from IDM/ACC (rules) on what the network predicts. Safety on top.

    A predict_speed model sets the ACC speed limit itself (learned from the expert's road-geometry speed);
    lead vehicles and red lights stay with the ACC rules, unless the model also perceives them (perceive): then the
    simulator's lead is replaced by the predicted one, a predicted red light or yield stops the car at its predicted
    line, and a predicted stop sign is served by rule (traffic law, like the IDM): halt within STOP_SIGN_ZONE_M for
    STOP_SIGN_HOLD_S, then go.
    """

    model: RouteSteeringNet
    speed_limit_mps: float = 13.9
    acc: ACCController = field(default_factory=ACCController)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    device: str = "cpu"
    name: str = "model"
    # a waypoints model steers by pure pursuit on its predicted path (a rule, the expert's controller and look-ahead)
    wheelbase_m: float = 3.0
    max_steer_rad: float = 0.5
    lookahead_gain: float = 0.5
    lookahead_offset_m: float = 2.0
    lookahead_min_m: float = 3.0

    def __post_init__(self) -> None:
        self.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kw) -> "ModelPolicy":
        ckpt = torch.load(path, map_location=kw.get("device", "cpu"), weights_only=False)
        cfg = ckpt.get("cfg", {})
        model = RouteSteeringNet(backbone=cfg.get("backbone", "small"), pretrained=False, predict_speed=cfg.get("predict_speed", False),
                                 perceive=cfg.get("perceive", False), light_camera=cfg.get("light_camera", False),
                                 surround=cfg.get("surround", False), waypoints=cfg.get("waypoints", False))
        model.load_state_dict(ckpt["model_state"])
        return cls(model, **kw)

    def reset(self) -> None:
        self._sign_served, self._sign_hold, self._sign_away, self._last_t = False, 0.0, 0.0, None
        self._red_on, self._red_clear, self._red_d = False, 0.0, PERCEPT_RANGE_M

    @torch.no_grad()
    def act(self, obs: Observation) -> Action:
        if obs.image is None:
            raise ValueError("ModelPolicy needs image observations")
        image = torch.from_numpy(np.array(obs.image, dtype=np.uint8)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        speed = torch.tensor([obs.speed_mps], dtype=torch.float32, device=self.device)
        command = torch.tensor([int(obs.command)], dtype=torch.int64, device=self.device)
        light = None
        if self.model.light_camera:
            if obs.extras.get("light_image") is None:
                raise ValueError("this model needs extras['light_image'] (the CARLA env's front-camera light crop)")
            light = torch.from_numpy(np.array(obs.extras["light_image"], dtype=np.uint8)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        surround = None
        if self.model.surround:
            views = obs.extras.get("surround_images") or {}
            if any(f not in views for f in SURROUND_FOLDERS):
                raise ValueError("this model needs extras['surround_images'] (the CARLA env's left/right/rear cameras)")
            # uint8 like the training loader: the model scales it to [0, 1] itself
            surround = torch.from_numpy(np.stack([views[f] for f in SURROUND_FOLDERS]).astype(np.uint8)).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        target_point = None
        if self.model.waypoints:
            if obs.extras.get("route_target") is None:
                raise ValueError("this model needs extras['route_target'] (the navigation target point)")
            target_point = torch.tensor([obs.extras["route_target"]], dtype=torch.float32, device=self.device)
        out = self.model(image, speed, command, light, surround, target_point)[0].cpu().numpy()
        steer = float(out[0])
        if self.model.waypoints:
            self.last_waypoints = [(float(x), float(y)) for x, y in out[-2 * N_WAYPOINTS:].reshape(-1, 2)]
            lookahead = min(20.0, max(self.lookahead_min_m, self.lookahead_gain * obs.speed_mps + self.lookahead_offset_m))
            steer = PurePursuit(self.wheelbase_m, lookahead, self.max_steer_rad).steer(self.last_waypoints)
        if not hasattr(self, "_sign_served"):
            self.reset()
        dt = 0.0 if self._last_t is None else max(obs.t - self._last_t, 0.0)
        self._last_t = obs.t
        stops: list[float] = []
        hold = False
        if self.model.perceive:
            p_red, red_d, sign_d, p_yield, yield_d, lead_d, lead_rel = self.last_percept = tuple(float(v) for v in out[4:11])
            # light state tracking (rule): one frame of 'not red' does not release a red; it must read clear for
            # RED_RELEASE_S (single-frame flips at the line let A run 21 reds, 11-01)
            self._red_clear = 0.0 if p_red > 0.5 else self._red_clear + dt
            if p_red > 0.5 or (self._red_on and self._red_clear < RED_RELEASE_S):
                self._red_on, self._red_d = True, red_d if p_red > 0.5 else self._red_d
                stops.append(self._red_d)
                # standstill hold (rule, like a production ACC at a stop): stopped at a red near its line, stay put
                # instead of creeping on a distance estimate (A crept from ~6 m into views it had never seen)
                hold = obs.speed_mps < HOLD_SPEED_MPS and self._red_d < HOLD_LINE_M
            else:
                self._red_on = False
            if p_yield > 0.5:
                stops.append(yield_d)
            stops += self._stop_sign(sign_d, obs.speed_mps, dt)
            lead = LeadInfo(lead_d, lead_rel) if lead_d < PERCEPT_RANGE_M - 2.0 else None
            radar = obs.extras.get("radar_lead")
            # late fusion (rule): the nearer of the camera's and the radar's in-path obstacle; the camera cannot see the
            # 0-4 m below the bonnet (12% of those leads seen), the radar can
            if radar is not None and (lead is None or radar.distance_m < lead.distance_m):
                lead = radar
            obs = replace(obs, lead=lead)
        limit = self.speed_limit_mps
        if self.model.predict_speed:
            limit = self.last_target_speed = min(limit, float(out[3]))
        accel = self.acc.accel(obs, limit)
        for d in stops:
            accel = min(accel, stop_accel(self.acc, obs.speed_mps, d))
        if hold:
            accel = -self.acc.max_brake_decel
        throttle, brake = self.acc.pedals(accel)
        return apply_safety(Action(steer, throttle, brake), obs, self.safety)

    def _stop_sign(self, sign_d: float, speed: float, dt: float) -> list[float]:
        """Stop line for a seen, not yet served stop sign (empty when none or served)."""
        if sign_d > STOP_SIGN_ZONE_M + 5.0:
            self._sign_away += dt
            if self._sign_away >= SIGN_RESET_S:
                self._sign_served, self._sign_hold = False, 0.0
        else:
            self._sign_away = 0.0
        if self._sign_served or sign_d >= PERCEPT_RANGE_M - 2.0:
            return []
        if sign_d <= STOP_SIGN_ZONE_M and speed < BLOCKED_SPEED_MPS:
            self._sign_hold += dt
            if self._sign_hold >= STOP_SIGN_HOLD_S:
                self._sign_served, self._sign_hold = True, 0.0
                return []
        return [sign_d]
