"""Simulated voluntary motor command (spec section 4.1).

    tau_vol = sat( tau_ff + K_h (q_ref - q) + D_h (qd_ref - qd) )

tau_ff is the inverse dynamics of the *reference* under the person's nominal arm model.
It contains no synergy term, so the simulated person is unaware of the impairment and
does not compensate for it by construction. Saturation acts on the total command, with
direction-specific limits per joint.

Sign convention: positive torque is the positive-angle direction. For the elbow that is
flexion, so tau_max_neg[1] is the maximum voluntary *extension* torque (tau_ext,max,e in
the spec), which is also the elbow's tau_scale for the impairment.
"""

from dataclasses import dataclass

import numpy as np

from .arm import ArmParams, coriolis_vector, mass_matrix
from .trajectory import Reference


@dataclass(frozen=True)
class HumanParams:
    K: np.ndarray  # feedback stiffness per joint (N m/rad), shape (2,)
    D: np.ndarray  # feedback damping per joint (N m s/rad), shape (2,)
    tau_max_pos: np.ndarray  # max torque in the positive-angle direction (N m), shape (2,)
    tau_max_neg: np.ndarray  # max torque magnitude in the negative-angle direction, shape (2,)

    def __post_init__(self):
        for name in ("K", "D", "tau_max_pos", "tau_max_neg"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or np.any(value <= 0.0):
                raise ValueError(f"{name} must have shape (2,) with positive entries")
            object.__setattr__(self, name, value)


def feedforward_torque(ref: Reference, nominal: ArmParams) -> np.ndarray:
    """Inverse dynamics of the reference under the nominal arm model."""
    return mass_matrix(ref.q, nominal) @ ref.qdd + coriolis_vector(ref.q, ref.qd, nominal) + nominal.b * ref.qd


def voluntary_torque(
    q: np.ndarray, qd: np.ndarray, ref: Reference, human: HumanParams, nominal: ArmParams
) -> np.ndarray:
    """Return the saturated voluntary joint torque for the current state and reference."""
    tau = feedforward_torque(ref, nominal) + human.K * (ref.q - q) + human.D * (ref.qd - qd)
    return np.clip(tau, -human.tau_max_neg, human.tau_max_pos)
