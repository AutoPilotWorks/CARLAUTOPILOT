from __future__ import annotations


class PID:
    def __init__(self, kp: float, ki: float, kd: float, out_min: float = -1.0, out_max: float = 1.0):
        if out_min >= out_max:
            raise ValueError("out_min must be below out_max")
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error: float | None = None

    def step(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")
        deriv = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        trial = self._integral + error * dt
        raw = self.kp * error + self.ki * trial + self.kd * deriv
        out = min(self.out_max, max(self.out_min, raw))
        saturated_same_dir = (raw > self.out_max and error > 0) or (raw < self.out_min and error < 0)
        if not saturated_same_dir:
            self._integral = trial
        return out
