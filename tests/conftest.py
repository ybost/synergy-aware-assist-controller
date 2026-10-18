import numpy as np
import pytest

from synergy_aware.arm import ArmParams


@pytest.fixture
def params() -> ArmParams:
    """Test-only parameters. These are NOT calibration values (spec section 2, TBD-CAL)."""
    return ArmParams(l1=0.30, l2=0.35, m1=2.0, m2=1.5, lc1=0.13, lc2=0.17, I1=0.02, I2=0.03, b=0.0)


@pytest.fixture
def damped_params(params: ArmParams) -> ArmParams:
    from dataclasses import replace

    return replace(params, b=0.2)


@pytest.fixture
def human_params():
    """Test-only human parameters. NOT calibration values (spec section 4.1, TBD-CAL)."""
    from synergy_aware.human import HumanParams

    return HumanParams(
        K=np.array([60.0, 40.0]),
        D=np.array([3.5, 2.7]),
        tau_max_pos=np.array([30.0, 30.0]),
        tau_max_neg=np.array([30.0, 25.0]),  # [1] is the elbow extension limit (tau_scale)
    )


@pytest.fixture
def robot_limits():
    """Test-only robot limits. NOT calibration values (charter section 14, TBD-CAL)."""
    from synergy_aware.controllers import RobotLimits

    return RobotLimits(tau_max=np.array([40.0, 30.0]), rate_max=np.array([2000.0, 2000.0]))


@pytest.fixture
def impedance_gains():
    """Test-only C1 gains. NOT calibration values."""
    from synergy_aware.controllers import ImpedanceGains

    return ImpedanceGains(Kp=np.array([120.0, 80.0]), Kd=np.array([6.0, 4.0]))


@pytest.fixture
def make_cfg(damped_params, human_params, robot_limits, reach_endpoints):
    """Factory for trial configs: make_cfg(alpha=..., u_abd=...)."""
    from synergy_aware.simulation import TrialConfig

    def _make(**overrides):
        base = dict(
            arm=damped_params,
            human=human_params,
            limits=robot_limits,
            x0=reach_endpoints[0],
            x1=reach_endpoints[1],
            T=1.0,
            t_end=3.0,
            dt=1e-3,
        )
        base.update(overrides)
        return TrialConfig(**base)

    return _make


@pytest.fixture
def reach_endpoints() -> tuple[np.ndarray, np.ndarray]:
    """A 0.2 m forward reach well inside the workspace, away from singularities."""
    return np.array([0.35, 0.15]), np.array([0.50, 0.30])


@pytest.fixture
def q_grid() -> list[np.ndarray]:
    """Postures spanning the working range, with the elbow flexed (q_e > 0)."""
    return [np.array([qs, qe]) for qs in np.linspace(-1.0, 1.0, 5) for qe in np.linspace(0.2, 2.2, 5)]
