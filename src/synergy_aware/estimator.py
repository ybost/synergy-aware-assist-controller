"""Online synergy estimator (spec section 6).

The estimator only ever sees an Observation. It deliberately has no field for the true
impairment torque or the true severity (the simulator's ground-truth channel).

Estimate (quasi-static elbow balance, flexion positive, e = q_e - q_ref,e):
    human torque   = -K_h e                    (feedback; feedforward matches the reference)
    robot torque   = tau_robot,e               (known: it is our own previous command)
    at rest        -K_h e + tau_robot,e + tau_syn = 0
    =>  d_hat = K_h * mean(e) - mean(tau_robot,e),   alpha_hat = d_hat / (u_abd * tau_scale)

Subtracting the robot's own torque keeps the estimate correct while a cancelling
controller is active; using only the residual would measure what is left over and
shrink as the cancellation works. The relation is quasi-static, so it is biased during
motion (inertial terms) and when the human command saturates.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    t: float
    q: np.ndarray  # measured joint angles (with simulated sensor noise)
    qd: np.ndarray  # measured or estimated joint velocities
    q_ref: np.ndarray  # planned reference joint angles
    qd_ref: np.ndarray  # planned reference joint velocities
    u_abd: float  # commanded shoulder-abduction demand
    tau_robot_prev: np.ndarray  # previous robot command


@dataclass(frozen=True)
class Estimate:
    e_hat: float  # windowed elbow residual (rad)
    alpha_hat: float | None  # None when undefined (u_abd <= u_min) or the window is not full
    valid: bool  # True once the window is full


class SynergyEstimator:
    def __init__(self, tau_scale: float, k_h_e: float, window: int, u_min: float, alpha_max: float = 1.0):
        if tau_scale <= 0.0 or k_h_e <= 0.0 or alpha_max <= 0.0 or u_min < 0.0:
            raise ValueError("tau_scale, k_h_e and alpha_max must be positive and u_min non-negative")
        if int(window) != window or window < 1:
            raise ValueError("window must be a positive integer")
        self.tau_scale = tau_scale
        self.k_h_e = k_h_e  # the estimator's nominal human elbow stiffness
        self.window = int(window)
        self.u_min = u_min
        self.alpha_max = alpha_max
        self._residual: deque[float] = deque(maxlen=self.window)
        self._robot_e: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._residual.clear()
        self._robot_e.clear()

    def update(self, obs: Observation) -> Estimate:
        self._residual.append(float(obs.q[1] - obs.q_ref[1]))
        self._robot_e.append(float(obs.tau_robot_prev[1]))
        e_hat = float(np.mean(self._residual))
        if len(self._residual) < self.window:
            return Estimate(e_hat=e_hat, alpha_hat=None, valid=False)
        if obs.u_abd <= self.u_min:
            return Estimate(e_hat=e_hat, alpha_hat=None, valid=True)
        d_hat = self.k_h_e * e_hat - float(np.mean(self._robot_e))
        alpha_hat = float(np.clip(d_hat / (obs.u_abd * self.tau_scale), 0.0, self.alpha_max))
        return Estimate(e_hat=e_hat, alpha_hat=alpha_hat, valid=True)
