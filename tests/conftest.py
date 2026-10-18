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
def q_grid() -> list[np.ndarray]:
    """Postures spanning the working range, with the elbow flexed (q_e > 0)."""
    return [np.array([qs, qe]) for qs in np.linspace(-1.0, 1.0, 5) for qe in np.linspace(0.2, 2.2, 5)]
