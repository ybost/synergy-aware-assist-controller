"""PLACEHOLDER parameters for exploratory runs.

These values are NOT calibrated and NOT sourced from the literature. They exist so the
pipeline can run end to end. Spec section 2 and the charter's calibration phase replace
them (anthropometrics, human gains, torque limits, severity levels, noise levels) before
any result is claimed. Anything produced with this module must be labelled exploratory.
"""

import numpy as np

from .arm import ArmParams
from .controllers import ImpedanceGains, RobotLimits
from .human import HumanParams
from .simulation import TrialConfig


def arm() -> ArmParams:
    return ArmParams(l1=0.30, l2=0.35, m1=2.0, m2=1.5, lc1=0.13, lc2=0.17, I1=0.02, I2=0.03, b=0.2)


def human() -> HumanParams:
    return HumanParams(
        K=np.array([60.0, 40.0]),
        D=np.array([3.5, 2.7]),
        tau_max_pos=np.array([30.0, 30.0]),
        tau_max_neg=np.array([30.0, 25.0]),  # [1] is the elbow extension limit, also tau_scale
    )


def limits() -> RobotLimits:
    return RobotLimits(tau_max=np.array([40.0, 30.0]), rate_max=np.array([2000.0, 2000.0]))


def c1_gains() -> ImpedanceGains:
    return ImpedanceGains(Kp=np.array([120.0, 80.0]), Kd=np.array([6.0, 4.0]))


def base_cfg(**overrides) -> TrialConfig:
    """A 0.2 m forward reach, 1 s duration, with the placeholder arm, person and limits."""
    cfg = dict(
        arm=arm(),
        human=human(),
        limits=limits(),
        x0=np.array([0.35, 0.15]),
        x1=np.array([0.50, 0.30]),
        T=1.0,
        t_end=2.5,
        dt=1e-3,
    )
    cfg.update(overrides)
    return TrialConfig(**cfg)
