"""Minimum-jerk reaching reference (spec section 3).

The hand follows a straight line from x0 to x1 with a minimum-jerk time profile. Joint
position, velocity and acceleration come from inverse kinematics and the Jacobian:
    qd  = J^-1 xd
    qdd = J^-1 (xdd - Jdot qd)
Before t = 0 the reference holds the start; after t = T it holds the target.
"""

from dataclasses import dataclass

import numpy as np

from .arm import ArmParams, ik, jacobian, jacobian_dot_qd

SINGULARITY_TOL = 1e-6


def min_jerk_profile(t: float, T: float) -> tuple[float, float, float]:
    """Return (s, ds/dt, d2s/dt2) for the minimum-jerk profile, with s from 0 to 1."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    if t <= 0.0:
        return 0.0, 0.0, 0.0
    if t >= T:
        return 1.0, 0.0, 0.0
    tau = t / T
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    return s, ds, dds


@dataclass(frozen=True)
class Reference:
    x: np.ndarray  # hand position (m)
    q: np.ndarray  # joint angles (rad)
    qd: np.ndarray  # joint velocities (rad/s)
    qdd: np.ndarray  # joint accelerations (rad/s^2)


class ReachReference:
    def __init__(self, x0: np.ndarray, x1: np.ndarray, T: float, p: ArmParams):
        self.x0 = np.asarray(x0, dtype=float)
        self.x1 = np.asarray(x1, dtype=float)
        self.T = float(T)
        self.p = p

    def at(self, t: float) -> Reference:
        s, ds, dds = min_jerk_profile(t, self.T)
        delta = self.x1 - self.x0
        x, xd, xdd = self.x0 + delta * s, delta * ds, delta * dds
        q = ik(x, self.p)
        J = jacobian(q, self.p)
        if abs(np.linalg.det(J)) < SINGULARITY_TOL:
            raise ValueError("reference passes through a kinematic singularity")
        qd = np.linalg.solve(J, xd)
        qdd = np.linalg.solve(J, xdd - jacobian_dot_qd(q, qd, self.p))
        return Reference(x=x, q=q, qd=qd, qdd=qdd)
