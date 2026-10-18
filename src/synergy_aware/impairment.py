"""Shoulder-abduction-dependent impairment torque (spec section 4.2). STUB."""

import numpy as np


def synergy_torque(alpha: float, u_abd: float, tau_scale: float) -> np.ndarray:
    """Return tau_syn = [0, alpha * u_abd * tau_scale] (elbow channel, flexion direction).

    alpha >= 0 is the severity; u_abd must lie in [0, 1]; tau_scale > 0 is the elbow's
    maximum voluntary extension torque. Raise ValueError for out-of-range inputs.
    """
    raise NotImplementedError
