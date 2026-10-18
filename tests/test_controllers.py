"""C0, C1 and the torque/torque-rate limits (spec section 5, charter section 14)."""

import numpy as np
import pytest

from synergy_aware.controllers import (
    ImpedanceController,
    ImpedanceGains,
    NullController,
    RobotLimits,
    limit_torque,
)
from synergy_aware.estimator import Observation


def _obs(q, qd, q_ref, qd_ref):
    return Observation(
        t=0.0,
        q=np.asarray(q, float),
        qd=np.asarray(qd, float),
        q_ref=np.asarray(q_ref, float),
        qd_ref=np.asarray(qd_ref, float),
        u_abd=0.5,
        tau_robot_prev=np.zeros(2),
    )


def test_null_controller_applies_no_torque():
    np.testing.assert_array_equal(NullController().torque(_obs([0, 1], [0, 0], [0.1, 1.2], [0, 0])), [0.0, 0.0])


def test_impedance_is_zero_on_the_reference(impedance_gains):
    ctrl = ImpedanceController(impedance_gains)
    np.testing.assert_allclose(ctrl.torque(_obs([0.3, 1.0], [0.2, -0.1], [0.3, 1.0], [0.2, -0.1])), 0.0, atol=1e-12)


def test_impedance_matches_pd_law_and_is_restoring(impedance_gains):
    ctrl = ImpedanceController(impedance_gains)
    tau = ctrl.torque(_obs([0.0, 0.0], [0.0, 0.0], [0.1, -0.2], [0.5, 0.25]))
    expected = impedance_gains.Kp * np.array([0.1, -0.2]) + impedance_gains.Kd * np.array([0.5, 0.25])
    np.testing.assert_allclose(tau, expected)
    assert tau[0] > 0.0 and tau[1] < 0.0  # pushes toward the reference


def test_impedance_ignores_the_abduction_demand(impedance_gains):
    """C1 is conventional: it must not use u_abd."""
    ctrl = ImpedanceController(impedance_gains)
    a = _obs([0, 1], [0, 0], [0.1, 1.2], [0, 0])
    b = Observation(**{**a.__dict__, "u_abd": 1.0})
    np.testing.assert_array_equal(ctrl.torque(a), ctrl.torque(b))


def test_limiter_passes_small_slow_commands_unchanged(robot_limits):
    tau = np.array([0.5, -0.4])
    np.testing.assert_allclose(limit_torque(tau, np.array([0.4, -0.35]), robot_limits, dt=1e-3), tau)


def test_limiter_enforces_magnitude(robot_limits):
    big = np.array([1e3, -1e3])
    out = limit_torque(big, robot_limits.tau_max * np.array([1, -1]) * 0.999, robot_limits, dt=1e-3)
    assert np.all(np.abs(out) <= robot_limits.tau_max + 1e-12)


def test_limiter_enforces_rate(robot_limits):
    dt = 1e-3
    out = limit_torque(np.array([30.0, 30.0]), np.zeros(2), robot_limits, dt)
    np.testing.assert_allclose(out, robot_limits.rate_max * dt)  # one step of allowed change only


def test_limiter_reaches_the_command_over_several_steps(robot_limits):
    dt, target, tau = 1e-3, np.array([5.0, -5.0]), np.zeros(2)
    for _ in range(10):
        tau = limit_torque(target, tau, robot_limits, dt)
    np.testing.assert_allclose(tau, target)


@pytest.mark.parametrize("bad", [np.array([0.0, 1.0]), np.array([-1.0, 1.0]), np.ones(3)])
def test_parameter_validation(bad):
    with pytest.raises(ValueError):
        RobotLimits(tau_max=bad, rate_max=np.ones(2))
    with pytest.raises(ValueError):
        ImpedanceGains(Kp=bad, Kd=np.ones(2))
