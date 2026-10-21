from __future__ import annotations

from dataclasses import dataclass

from carlautopilot.core import Action, Observation


@dataclass(frozen=True)
class SafetyConfig:
    ttc_brake_s: float = 2.5
    ttc_full_brake_s: float = 1.0


def time_to_collision(obs: Observation) -> float | None:
    if obs.lead is None or obs.lead.rel_speed_mps >= 0:
        return None
    return obs.lead.distance_m / -obs.lead.rel_speed_mps


def apply_safety(action: Action, obs: Observation, cfg: SafetyConfig = SafetyConfig()) -> Action:
    ttc = time_to_collision(obs)
    if ttc is None or ttc >= cfg.ttc_brake_s:
        return action.clipped()
    span = max(cfg.ttc_brake_s - cfg.ttc_full_brake_s, 1e-6)
    level = min(1.0, max(0.0, (cfg.ttc_brake_s - ttc) / span))
    return Action(action.steer, 0.0, max(action.brake, level)).clipped()
