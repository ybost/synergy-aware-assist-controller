"""Shoulder-abduction-dependent impairment torque (spec section 4.2)."""

import numpy as np


def synergy_torque(alpha: float, u_abd: float, tau_scale: float) -> np.ndarray:
    """Return tau_syn = [0, alpha * u_abd * tau_scale] (elbow channel, flexion direction).

    alpha >= 0 is the severity; u_abd must lie in [0, 1]; tau_scale > 0 is the elbow's
    maximum voluntary extension torque. Raise ValueError for out-of-range inputs.

    Initial linear load response f(u) = u (charter section 10): an engineering
    abstraction, not a claimed physiological law.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if not 0.0 <= u_abd <= 1.0:
        raise ValueError(f"u_abd must lie in [0, 1], got {u_abd}")
    if tau_scale <= 0.0:
        raise ValueError(f"tau_scale must be > 0, got {tau_scale}")
    return np.array([0.0, alpha * u_abd * tau_scale])
