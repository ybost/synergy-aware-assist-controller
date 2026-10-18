"""Closed-loop trials: person + impairment + robot controller."""

import numpy as np
import pytest

from synergy_aware.controllers import ImpedanceController, NullController
from synergy_aware.impairment import synergy_torque
from synergy_aware.simulation import run_trial


def test_unimpaired_c0_reaches_the_target(make_cfg):
    assert run_trial(make_cfg(), NullController()).endpoint_error() < 1e-3


def test_ground_truth_is_recorded_but_matches_the_impairment_model(make_cfg, human_params):
    cfg = make_cfg(alpha=0.3, u_abd=0.5)
    res = run_trial(cfg, NullController())
    expected = synergy_torque(0.3, 0.5, human_params.tau_max_neg[1])
    np.testing.assert_allclose(res.tau_syn, np.tile(expected, (res.tau_syn.shape[0], 1)))


def test_controller_never_sees_ground_truth(make_cfg):
    seen = []

    class Spy:
        def torque(self, obs):
            seen.append(obs)
            return np.zeros(2)

    run_trial(make_cfg(alpha=0.5, u_abd=1.0, t_end=0.05), Spy())
    assert seen
    for obs in seen:
        assert not {"tau_syn", "alpha"} & set(vars(obs))


def test_c1_does_not_degrade_unimpaired_tracking(make_cfg, impedance_gains):
    c0 = run_trial(make_cfg(), NullController())
    c1 = run_trial(make_cfg(), ImpedanceController(impedance_gains))
    assert c1.endpoint_error() <= c0.endpoint_error() + 1e-4


def test_static_elbow_offset_matches_analytic_stiffness_balance(make_cfg, impedance_gains, human_params):
    """At rest, (K_h + Kp) * offset = tau_syn on the elbow, so the robot shrinks the offset."""
    alpha, u = 0.04, 1.0
    tau_syn_e = synergy_torque(alpha, u, human_params.tau_max_neg[1])[1]
    c0 = run_trial(make_cfg(alpha=alpha, u_abd=u), NullController())
    c1 = run_trial(make_cfg(alpha=alpha, u_abd=u), ImpedanceController(impedance_gains))
    assert c0.final_elbow_offset() == pytest.approx(tau_syn_e / human_params.K[1], rel=0.02)
    assert c1.final_elbow_offset() == pytest.approx(tau_syn_e / (human_params.K[1] + impedance_gains.Kp[1]), rel=0.02)
    assert 0.0 < c1.final_elbow_offset() < c0.final_elbow_offset()


def test_recorded_robot_torque_respects_limits(make_cfg, impedance_gains, robot_limits):
    cfg = make_cfg(alpha=0.5, u_abd=1.0, t_end=1.5)
    res = run_trial(cfg, ImpedanceController(impedance_gains))
    assert np.all(np.abs(res.tau_robot) <= robot_limits.tau_max + 1e-12)
    steps = np.abs(np.diff(res.tau_robot, axis=0, prepend=np.zeros((1, 2))))
    assert np.all(steps <= robot_limits.rate_max * cfg.dt + 1e-12)


def test_trials_are_deterministic(make_cfg, impedance_gains):
    a = run_trial(make_cfg(alpha=0.2, u_abd=0.7, t_end=0.5), ImpedanceController(impedance_gains))
    b = run_trial(make_cfg(alpha=0.2, u_abd=0.7, t_end=0.5), ImpedanceController(impedance_gains))
    np.testing.assert_array_equal(a.q, b.q)
