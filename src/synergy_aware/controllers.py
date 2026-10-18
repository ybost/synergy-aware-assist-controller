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


class ElbowBoostController:
    """C2: base controller plus always-on extra elbow stiffness and damping.

    Generic assistance: no detection and no use of u_abd. The extra gains are set by the
    budget-matching rule in matching.py.
    """

    def __init__(self, base: Controller, dK: float, dD: float):
        if dK < 0.0 or dD < 0.0:
            raise ValueError("dK and dD must be non-negative")
        self.base = base
        self.dK = dK
        self.dD = dD

    def torque(self, obs: Observation) -> np.ndarray:
        tau = np.array(self.base.torque(obs), dtype=float)
        tau[1] += self.dK * (obs.q_ref[1] - obs.q[1]) + self.dD * (obs.qd_ref[1] - obs.qd[1])
        return tau


def _elbow_torque(alpha: float, u_abd: float, tau_scale: float) -> np.ndarray:
    """Elbow-extension torque that cancels a synergy of severity alpha at demand u_abd."""
    return np.array([0.0, -alpha * u_abd * tau_scale])


class SynergyAwareController:
    """C3: base controller plus estimated-synergy cancellation, -alpha_hat * u_abd * tau_scale.

    The estimator is stateful; run_trial calls reset() before each trial.

    ramp_steps > 0 ramps the cancellation in linearly over that many control steps from
    the first usable estimate (and restarts the ramp whenever the estimate becomes
    unusable), instead of stepping it on. ramp_steps = 0 applies it at full strength
    immediately.
    """

    def __init__(self, base: Controller, estimator, tau_scale: float, ramp_steps: int = 0):
        if int(ramp_steps) != ramp_steps or ramp_steps < 0:
            raise ValueError("ramp_steps must be a non-negative integer")
        self.base = base
        self.estimator = estimator
        self.tau_scale = tau_scale
        self.ramp_steps = int(ramp_steps)
        self.estimates: list = []
        self._active_steps = 0

    def reset(self) -> None:
        self.estimator.reset()
        self.estimates = []
        self._active_steps = 0

    def torque(self, obs: Observation) -> np.ndarray:
        est = self.estimator.update(obs)
        self.estimates.append(est)
        tau = np.asarray(self.base.torque(obs), dtype=float)
        if est.valid and est.alpha_hat is not None:
            self._active_steps += 1
            gain = 1.0 if self.ramp_steps == 0 else min(1.0, self._active_steps / self.ramp_steps)
            tau = tau + gain * _elbow_torque(est.alpha_hat, obs.u_abd, self.tau_scale)
        else:
            self._active_steps = 0
        return tau


class OracleController:
    """C-oracle: base controller plus cancellation using the TRUE alpha.

    Uses ground truth, so it is not a realistic controller. It is an upper bound only and
    must be labelled as such wherever it appears.
    """

    def __init__(self, base: Controller, alpha_true: float, tau_scale: float):
        self.base = base
        self.alpha_true = alpha_true
        self.tau_scale = tau_scale

    def torque(self, obs: Observation) -> np.ndarray:
        return np.asarray(self.base.torque(obs), dtype=float) + _elbow_torque(
            self.alpha_true, obs.u_abd, self.tau_scale
        )


def limit_torque(tau: np.ndarray, tau_prev: np.ndarray, limits: RobotLimits, dt: float) -> np.ndarray:
    """Rate-limit the change from the previous command, then saturate the magnitude."""
    step = np.clip(tau - tau_prev, -limits.rate_max * dt, limits.rate_max * dt)
    return np.clip(tau_prev + step, -limits.tau_max, limits.tau_max)
