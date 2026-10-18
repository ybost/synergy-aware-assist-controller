"""Robot controllers (spec section 5): C0 unassisted and C1 conventional impedance.

Every controller maps an Observation (measured state, reference, commanded u_abd and the
previous robot command) to a raw joint torque. The trial runner then applies the torque
and torque-rate limits (charter section 14) to whatever a controller returns.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .estimator import Observation


def _positive_pair(name: str, value) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or np.any(value <= 0.0):
        raise ValueError(f"{name} must have shape (2,) with positive entries")
    return value


@dataclass(frozen=True)
class RobotLimits:
    tau_max: np.ndarray  # torque magnitude limit per joint (N m)
    rate_max: np.ndarray  # torque-rate limit per joint (N m/s)

    def __post_init__(self):
        object.__setattr__(self, "tau_max", _positive_pair("tau_max", self.tau_max))
        object.__setattr__(self, "rate_max", _positive_pair("rate_max", self.rate_max))


@dataclass(frozen=True)
class ImpedanceGains:
    Kp: np.ndarray  # stiffness per joint (N m/rad)
    Kd: np.ndarray  # damping per joint (N m s/rad)

    def __post_init__(self):
        object.__setattr__(self, "Kp", _positive_pair("Kp", self.Kp))
        object.__setattr__(self, "Kd", _positive_pair("Kd", self.Kd))


class Controller(Protocol):
    def torque(self, obs: Observation) -> np.ndarray: ...


class NullController:
    """C0: no robot assistance."""

    def torque(self, obs: Observation) -> np.ndarray:
        return np.zeros(2)


class ImpedanceController:
    """C1: tau_base = Kp (q_ref - q) + Kd (qd_ref - qd)."""

    def __init__(self, gains: ImpedanceGains):
        self.gains = gains

    def torque(self, obs: Observation) -> np.ndarray:
        return self.gains.Kp * (obs.q_ref - obs.q) + self.gains.Kd * (obs.qd_ref - obs.qd)


def limit_torque(tau: np.ndarray, tau_prev: np.ndarray, limits: RobotLimits, dt: float) -> np.ndarray:
    """Rate-limit the change from the previous command, then saturate the magnitude."""
    step = np.clip(tau - tau_prev, -limits.rate_max * dt, limits.rate_max * dt)
    return np.clip(tau_prev + step, -limits.tau_max, limits.tau_max)
