"""Two-link planar arm dynamics (spec section 2). STUB: signatures only.

Conventions (spec section 1): SI units, radians. q = [q_s, q_e]; q_e is the elbow
angle relative to the upper arm, flexion positive, q_e = 0 is full extension.
Dynamics: M(q) qdd + C(q, qd) qd + B qd = tau, with B = b * I (viscous, per joint).
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
    raise NotImplementedError


def coriolis_vector(q: np.ndarray, qd: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the Coriolis/centrifugal vector C(q, qd) qd, shape (2,)."""
    raise NotImplementedError


def kinetic_energy(q: np.ndarray, qd: np.ndarray, p: ArmParams) -> float:
    """Return 0.5 * qd^T M(q) qd."""
    raise NotImplementedError


def forward_dynamics(q: np.ndarray, qd: np.ndarray, tau: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return qdd = M^-1 (tau - C qd - b qd)."""
    raise NotImplementedError


def fk(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the hand position [x, y] (m)."""
    raise NotImplementedError


def ik(x: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return q for hand position x, choosing the elbow-flexed solution (q_e >= 0)."""
    raise NotImplementedError


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
    with shapes (n,), (n, 2), (n, 2), including the initial state."""
    raise NotImplementedError
