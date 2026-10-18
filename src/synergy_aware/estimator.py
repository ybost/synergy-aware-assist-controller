"""Online synergy estimator (spec section 6). STUB.

The estimator only ever sees an Observation. It deliberately has no field for the true
impairment torque or the true severity (the simulator's ground-truth channel).
"""

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
    e_hat: float  # windowed elbow residual
    alpha_hat: float | None  # None when undefined (u_abd <= u_min) or invalid
    valid: bool


class SynergyEstimator:
    def __init__(self, tau_scale: float, k_eff_e: float, window: int, u_min: float):
        raise NotImplementedError

    def update(self, obs: Observation) -> Estimate:
        raise NotImplementedError
