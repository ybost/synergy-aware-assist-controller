"""Sensor-noise model in the trial runner."""

import numpy as np
import pytest

from synergy_aware.controllers import ImpedanceController, NullController
from synergy_aware.simulation import NoiseConfig, qd_sd_from_differentiation, run_trial


def test_validation():
    with pytest.raises(ValueError):
        NoiseConfig(q_sd=-1e-3)
    with pytest.raises(ValueError):
        NoiseConfig(qd_sd=-1.0)


def test_differentiation_helper():
    assert qd_sd_from_differentiation(1e-3, 1e-3) == pytest.approx(np.sqrt(2.0))


def test_zero_noise_matches_noise_free_trial(make_cfg, impedance_gains):
    ctl = ImpedanceController(impedance_gains)
    a = run_trial(make_cfg(alpha=0.2, u_abd=1.0, t_end=1.0), ctl)
    b = run_trial(make_cfg(alpha=0.2, u_abd=1.0, t_end=1.0, noise=NoiseConfig()), ctl)
    np.testing.assert_array_equal(a.q, b.q)


def test_same_seed_reproduces_and_different_seed_differs(make_cfg, impedance_gains):
    ctl = ImpedanceController(impedance_gains)
    cfg = lambda seed: make_cfg(alpha=0.2, u_abd=1.0, t_end=1.0, noise=NoiseConfig(1e-3, 0.5, seed))  # noqa: E731
    np.testing.assert_array_equal(run_trial(cfg(1), ctl).q, run_trial(cfg(1), ctl).q)
    assert not np.array_equal(run_trial(cfg(1), ctl).q, run_trial(cfg(2), ctl).q)


def test_noise_is_identical_across_controllers(make_cfg, impedance_gains):
    """Common random numbers: the injected noise cannot depend on what the controller does."""
    cfg = make_cfg(alpha=0.2, u_abd=1.0, t_end=1.0, noise=NoiseConfig(1e-3, 0.5, seed=7))
    a = run_trial(cfg, NullController())
    b = run_trial(cfg, ImpedanceController(impedance_gains))
    assert not np.allclose(a.q, b.q)  # the trajectories differ ...
    np.testing.assert_allclose((a.q_meas - a.q[:-1]), (b.q_meas - b.q[:-1]), atol=1e-12)  # ... the noise does not
    np.testing.assert_allclose((a.qd_meas - a.qd[:-1]), (b.qd_meas - b.qd[:-1]), atol=1e-12)


def test_noise_statistics(make_cfg):
    res = run_trial(make_cfg(t_end=3.0, noise=NoiseConfig(q_sd=2e-3, qd_sd=0.7, seed=3)), NullController())
    dq, dqd = res.q_meas - res.q[:-1], res.qd_meas - res.qd[:-1]
    assert dq.std() == pytest.approx(2e-3, rel=0.05)
    assert dqd.std() == pytest.approx(0.7, rel=0.05)
    assert abs(dq.mean()) < 2e-4 and abs(dqd.mean()) < 0.05


def test_controller_observes_the_measured_state(make_cfg):
    seen = []

    class Spy:
        def torque(self, obs):
            seen.append((obs.q.copy(), obs.qd.copy()))
            return np.zeros(2)

    res = run_trial(make_cfg(t_end=0.1, noise=NoiseConfig(1e-3, 0.5, seed=5)), Spy())
    np.testing.assert_array_equal(np.array([s[0] for s in seen]), res.q_meas)
    np.testing.assert_array_equal(np.array([s[1] for s in seen]), res.qd_meas)


def test_person_and_metrics_use_the_true_state(make_cfg):
    """A controller that ignores its observation must be unaffected by sensor noise."""
    clean = run_trial(make_cfg(alpha=0.2, u_abd=1.0, t_end=1.0), NullController())
    noisy = run_trial(make_cfg(alpha=0.2, u_abd=1.0, t_end=1.0, noise=NoiseConfig(5e-3, 2.0, 1)), NullController())
    np.testing.assert_array_equal(clean.q, noisy.q)


def test_noise_degrades_a_feedback_controller(make_cfg, impedance_gains):
    ctl = ImpedanceController(impedance_gains)
    clean = run_trial(make_cfg(t_end=1.5), ctl)
    noisy = run_trial(make_cfg(t_end=1.5, noise=NoiseConfig(2e-3, 1.0, 1)), ctl)
    assert not np.array_equal(clean.tau_robot, noisy.tau_robot)
    assert noisy.tau_robot.std() > clean.tau_robot.std()
