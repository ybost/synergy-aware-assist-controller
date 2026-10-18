"""Two-link planar arm dynamics (spec section 2).

Conventions (spec section 1): SI units, radians. q = [q_s, q_e]; q_e is the elbow
angle relative to the upper arm, flexion positive, q_e = 0 is full extension.
Dynamics: M(q) qdd + C(q, qd) qd + B qd = tau, with B = b * I (viscous, per joint).
Horizontal plane, so there is no gravity term.

C(q, qd) qd uses the standard two-link form with h = m2 l1 lc2 sin(q_e):
    C(q, qd) qd = [-h qe_d (2 qs_d + qe_d),  h qs_d^2]
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmParams:
    l1: float  # upper-arm length (m)
    l2: float  # forearm length (m)
    m1: float  # upper-arm mass (kg)
    m2: float  # forearm + hand mass (kg)
    lc1: float  # shoulder to upper-arm centre of mass (m)
    lc2: float  # elbow to forearm centre of mass (m)
    I1: float  # upper-arm inertia about its centre of mass (kg m^2)
    I2: float  # forearm inertia about its centre of mass (kg m^2)
    b: float = 0.0  # viscous damping coefficient per joint (N m s/rad)


def mass_matrix(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the symmetric positive-definite inertia matrix M(q), shape (2, 2)."""
    c2 = np.cos(q[1])
    m22 = p.I2 + p.m2 * p.lc2**2
    m12 = m22 + p.m2 * p.l1 * p.lc2 * c2
    m11 = p.I1 + p.m1 * p.lc1**2 + p.m2 * p.l1**2 + m22 + 2.0 * p.m2 * p.l1 * p.lc2 * c2
    return np.array([[m11, m12], [m12, m22]])


def coriolis_vector(q: np.ndarray, qd: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the Coriolis/centrifugal vector C(q, qd) qd, shape (2,)."""
    h = p.m2 * p.l1 * p.lc2 * np.sin(q[1])
    return np.array([-h * qd[1] * (2.0 * qd[0] + qd[1]), h * qd[0] ** 2])


def kinetic_energy(q: np.ndarray, qd: np.ndarray, p: ArmParams) -> float:
    """Return 0.5 * qd^T M(q) qd."""
    return float(0.5 * qd @ mass_matrix(q, p) @ qd)


def forward_dynamics(q: np.ndarray, qd: np.ndarray, tau: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return qdd = M^-1 (tau - C qd - b qd)."""
    rhs = tau - coriolis_vector(q, qd, p) - p.b * qd
    return np.linalg.solve(mass_matrix(q, p), rhs)


def fk(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the hand position [x, y] (m)."""
    a1, a12 = q[0], q[0] + q[1]
    return np.array([p.l1 * np.cos(a1) + p.l2 * np.cos(a12), p.l1 * np.sin(a1) + p.l2 * np.sin(a12)])


def ik(x: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return q for hand position x, choosing the elbow-flexed solution (q_e >= 0).

    Raise ValueError if x is outside the reachable annulus.
    """
    r2 = float(x[0] ** 2 + x[1] ** 2)
    cos_e = (r2 - p.l1**2 - p.l2**2) / (2.0 * p.l1 * p.l2)
    if abs(cos_e) > 1.0 + 1e-9:
        raise ValueError(f"target {x} is outside the reachable workspace")
    q_e = float(np.arccos(np.clip(cos_e, -1.0, 1.0)))
    q_s = float(np.arctan2(x[1], x[0]) - np.arctan2(p.l2 * np.sin(q_e), p.l1 + p.l2 * np.cos(q_e)))
    return np.array([q_s, q_e])


def jacobian(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the hand Jacobian J(q) with xdot = J qd, shape (2, 2)."""
    a1, a12 = q[0], q[0] + q[1]
    s1, c1 = np.sin(a1), np.cos(a1)
    s12, c12 = np.sin(a12), np.cos(a12)
    return np.array([[-p.l1 * s1 - p.l2 * s12, -p.l2 * s12], [p.l1 * c1 + p.l2 * c12, p.l2 * c12]])


def jacobian_dot_qd(q: np.ndarray, qd: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return (dJ/dt) qd, the velocity-dependent part of the hand acceleration."""
    a1, a12 = q[0], q[0] + q[1]
    s1, c1 = np.sin(a1), np.cos(a1)
    s12, c12 = np.sin(a12), np.cos(a12)
    dJ_dqs = np.array([[-p.l1 * c1 - p.l2 * c12, -p.l2 * c12], [-p.l1 * s1 - p.l2 * s12, -p.l2 * s12]])
    dJ_dqe = np.array([[-p.l2 * c12, -p.l2 * c12], [-p.l2 * s12, -p.l2 * s12]])
    return (dJ_dqs * qd[0] + dJ_dqe * qd[1]) @ qd


TorqueFn = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def integrate(
    q0: np.ndarray,
    qd0: np.ndarray,
    tau_fn: TorqueFn,
    p: ArmParams,
    dt: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-step RK4 integration. tau_fn(t, q, qd) -> tau. Return (t, q, qd) as arrays
    with shapes (n,), (n, 2), (n, 2), including the initial state.

    tau_fn is evaluated at each RK4 stage, so the torque may depend on the stage state.
    """
    if dt <= 0.0 or T < 0.0:
        raise ValueError("dt must be positive and T non-negative")
    n = int(round(T / dt))
    t = np.arange(n + 1) * dt
    q = np.empty((n + 1, 2))
    qd = np.empty((n + 1, 2))
    q[0], qd[0] = q0, qd0

    def deriv(ti: float, qi: np.ndarray, qdi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return qdi, forward_dynamics(qi, qdi, np.asarray(tau_fn(ti, qi, qdi), dtype=float), p)

    for k in range(n):
        tk, qk, qdk = t[k], q[k], qd[k]
        k1q, k1v = deriv(tk, qk, qdk)
        k2q, k2v = deriv(tk + dt / 2, qk + dt / 2 * k1q, qdk + dt / 2 * k1v)
        k3q, k3v = deriv(tk + dt / 2, qk + dt / 2 * k2q, qdk + dt / 2 * k2v)
        k4q, k4v = deriv(tk + dt, qk + dt * k3q, qdk + dt * k3v)
        q[k + 1] = qk + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        qd[k + 1] = qdk + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return t, q, qd
