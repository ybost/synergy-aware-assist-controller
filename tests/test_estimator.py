"""Estimator and the C3 / oracle controllers (spec sections 5 and 6)."""

import numpy as np
import pytest

from synergy_aware.controllers import (
    ImpedanceController,
    NullController,
    OracleController,
    SynergyAwareController,
)
from synergy_aware.estimator import Estimate, Observation, SynergyEstimator
from synergy_aware.metrics import ReferenceSet, excess_elbow_flexion
from synergy_aware.simulation import run_trial

TAU_SCALE, K_H_E = 25.0, 40.0


def _obs(e=0.0, tau_robot_e=0.0, u=1.0, t=0.0):
    return Observation(
        t=t,
        q=np.array([0.3, 1.0 + e]),
        qd=np.zeros(2),
        q_ref=np.array([0.3, 1.0]),
        qd_ref=np.zeros(2),
        u_abd=u,
        tau_robot_prev=np.array([0.0, tau_robot_e]),
    )


def _estimator(window=5, u_min=0.05, alpha_max=1.0):
    return SynergyEstimator(tau_scale=TAU_SCALE, k_h_e=K_H_E, window=window, u_min=u_min, alpha_max=alpha_max)


def _feed(est, n, **kw):
    out = None
    for _ in range(n):
        out = est.update(_obs(**kw))
    return out


def test_invalid_until_the_window_is_full():
    est = _estimator(window=5)
    for _ in range(4):
        out = est.update(_obs(e=0.01))
        assert out.valid is False and out.alpha_hat is None
    assert est.update(_obs(e=0.01)).valid is True


def test_static_balance_formula_without_robot_torque():
    out = _feed(_estimator(), 5, e=0.025, tau_robot_e=0.0, u=1.0)
    assert out.alpha_hat == pytest.approx(K_H_E * 0.025 / TAU_SCALE)  # = 0.04


def test_robot_torque_is_accounted_for():
    """With the same measured residual, a restoring robot torque means a larger synergy."""
    out = _feed(_estimator(), 5, e=0.01, tau_robot_e=-0.6, u=1.0)  # robot pushes toward extension
    assert out.alpha_hat == pytest.approx((K_H_E * 0.01 + 0.6) / TAU_SCALE)


def test_alpha_hat_scales_inversely_with_u_abd():
    hi = _feed(_estimator(), 5, e=0.01, u=1.0).alpha_hat
    lo = _feed(_estimator(), 5, e=0.01, u=0.5).alpha_hat
    assert lo == pytest.approx(2 * hi)


def test_undefined_at_or_below_u_min_but_residual_still_reported():
    out = _feed(_estimator(u_min=0.05), 5, e=0.02, u=0.05)
    assert out.valid and out.alpha_hat is None
    assert out.e_hat == pytest.approx(0.02)


def test_clipped_to_the_allowed_range():
    assert _feed(_estimator(), 5, e=-0.05).alpha_hat == 0.0  # negative disturbance clips to zero
    assert _feed(_estimator(alpha_max=0.5), 5, e=0.5).alpha_hat == 0.5


def test_window_is_causal_moving_average():
    est = _estimator(window=3)
    es = [0.0, 0.03, 0.06, 0.09]
    outs = [est.update(_obs(e=e)) for e in es]
    assert outs[2].e_hat == pytest.approx(np.mean(es[0:3]))
    assert outs[3].e_hat == pytest.approx(np.mean(es[1:4]))


def test_reset_clears_state():
    est = _estimator()
    _feed(est, 5, e=0.02)
    est.reset()
    assert est.update(_obs(e=0.02)).valid is False


@pytest.mark.parametrize("kwargs", [dict(window=0), dict(window=2.5), dict(alpha_max=0.0), dict(u_min=-0.1)])
def test_constructor_validation(kwargs):
    base = dict(tau_scale=TAU_SCALE, k_h_e=K_H_E, window=5, u_min=0.05)
    with pytest.raises(ValueError):
        SynergyEstimator(**{**base, **kwargs})


# --- controllers (duck-typed fake estimator) -----------------------------------------------


class _FakeEstimator:
    def __init__(self, estimate):
        self.estimate, self.resets = estimate, 0

    def update(self, obs):
        return self.estimate

    def reset(self):
        self.resets += 1


def test_c3_adds_extension_torque_on_the_elbow_only(impedance_gains):
    base = ImpedanceController(impedance_gains)
    obs = _obs(e=0.0, u=0.8)
    c3 = SynergyAwareController(base, _FakeEstimator(Estimate(0.0, 0.3, True)), tau_scale=TAU_SCALE)
    np.testing.assert_allclose(c3.torque(obs) - base.torque(obs), [0.0, -0.3 * 0.8 * TAU_SCALE])


@pytest.mark.parametrize("est", [Estimate(0.0, None, True), Estimate(0.0, 0.3, False)])
def test_c3_falls_back_to_the_base_controller_when_estimate_unusable(impedance_gains, est):
    base = ImpedanceController(impedance_gains)
    obs = _obs(e=0.01, u=0.8)
    c3 = SynergyAwareController(base, _FakeEstimator(est), tau_scale=TAU_SCALE)
    np.testing.assert_array_equal(c3.torque(obs), base.torque(obs))


def test_c3_ramps_the_cancellation_in_linearly(impedance_gains):
    base = ImpedanceController(impedance_gains)
    obs = _obs(u=1.0)
    c3 = SynergyAwareController(base, _FakeEstimator(Estimate(0.0, 0.4, True)), TAU_SCALE, ramp_steps=4)
    full = 0.4 * TAU_SCALE
    extra = [(c3.torque(obs) - base.torque(obs))[1] for _ in range(6)]
    np.testing.assert_allclose(extra, [-full * f for f in (0.25, 0.5, 0.75, 1.0, 1.0, 1.0)])


def test_c3_ramp_restarts_when_the_estimate_becomes_unusable(impedance_gains):
    base = ImpedanceController(impedance_gains)
    fake = _FakeEstimator(Estimate(0.0, 0.4, True))
    c3 = SynergyAwareController(base, fake, TAU_SCALE, ramp_steps=2)
    for _ in range(3):
        c3.torque(_obs())
    fake.estimate = Estimate(0.0, None, True)
    np.testing.assert_array_equal(c3.torque(_obs()), base.torque(_obs()))
    fake.estimate = Estimate(0.0, 0.4, True)
    assert (c3.torque(_obs()) - base.torque(_obs()))[1] == pytest.approx(-0.4 * TAU_SCALE * 0.5)  # ramp began again


def test_c3_reset_restarts_the_ramp(impedance_gains):
    base = ImpedanceController(impedance_gains)
    c3 = SynergyAwareController(base, _FakeEstimator(Estimate(0.0, 0.4, True)), TAU_SCALE, ramp_steps=2)
    c3.torque(_obs())
    c3.torque(_obs())
    c3.reset()
    assert (c3.torque(_obs()) - base.torque(_obs()))[1] == pytest.approx(-0.4 * TAU_SCALE * 0.5)


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_c3_rejects_invalid_ramp(impedance_gains, bad):
    with pytest.raises(ValueError):
        SynergyAwareController(ImpedanceController(impedance_gains), _FakeEstimator(None), TAU_SCALE, ramp_steps=bad)


def test_oracle_uses_the_true_alpha(impedance_gains):
    base = ImpedanceController(impedance_gains)
    obs = _obs(u=0.5)
    oracle = OracleController(base, alpha_true=0.2, tau_scale=TAU_SCALE)
    np.testing.assert_allclose(oracle.torque(obs) - base.torque(obs), [0.0, -0.2 * 0.5 * TAU_SCALE])


# --- closed loop -----------------------------------------------------------------------------


class _Probe:
    """Wraps a controller and an estimator; applies the controller, records the estimates."""

    def __init__(self, inner, estimator):
        self.inner, self.estimator, self.estimates = inner, estimator, []

    def reset(self):
        self.estimator.reset()
        self.estimates = []

    def torque(self, obs):
        self.estimates.append(self.estimator.update(obs))
        return self.inner.torque(obs)


@pytest.mark.parametrize("which", ["c0", "c1"])
def test_estimate_converges_to_true_alpha_for_open_loop_controllers(make_cfg, impedance_gains, human_params, which):
    alpha = 0.04
    inner = NullController() if which == "c0" else ImpedanceController(impedance_gains)
    probe = _Probe(inner, _estimator(window=100))
    run_trial(make_cfg(alpha=alpha, u_abd=1.0), probe)
    assert probe.estimates[-1].alpha_hat == pytest.approx(alpha, rel=0.03)


def _c3(impedance_gains, human_params, window=100):
    est = SynergyEstimator(
        tau_scale=human_params.tau_max_neg[1], k_h_e=human_params.K[1], window=window, u_min=0.05
    )
    return SynergyAwareController(ImpedanceController(impedance_gains), est, human_params.tau_max_neg[1])


def test_c3_closed_loop_estimate_stays_correct_while_cancelling(make_cfg, impedance_gains, human_params):
    """The estimate must not collapse as the cancellation removes the residual."""
    alpha = 0.2
    c3 = _c3(impedance_gains, human_params)
    run_trial(make_cfg(alpha=alpha, u_abd=1.0, t_end=4.0), c3)
    assert c3.estimates[-1].alpha_hat == pytest.approx(alpha, rel=0.05)


def test_ordering_oracle_c3_c1_c0_on_excess_flexion(make_cfg, impedance_gains, human_params):
    alpha, tau_scale = 0.2, human_params.tau_max_neg[1]
    cfg = make_cfg(alpha=alpha, u_abd=1.0, t_end=4.0)
    c1 = ImpedanceController(impedance_gains)
    ref_qe = ReferenceSet(c1).get(cfg)
    e = {
        "c0": excess_elbow_flexion(run_trial(cfg, NullController()), ref_qe),
        "c1": excess_elbow_flexion(run_trial(cfg, c1), ref_qe),
        "c3": excess_elbow_flexion(run_trial(cfg, _c3(impedance_gains, human_params)), ref_qe),
        "oracle": excess_elbow_flexion(run_trial(cfg, OracleController(c1, alpha, tau_scale)), ref_qe),
    }
    assert e["oracle"] < e["c3"] < e["c1"] < e["c0"]


def test_oracle_cancels_the_static_offset(make_cfg, impedance_gains, human_params):
    alpha, tau_scale = 0.2, human_params.tau_max_neg[1]
    res = run_trial(
        make_cfg(alpha=alpha, u_abd=1.0), OracleController(ImpedanceController(impedance_gains), alpha, tau_scale)
    )
    assert abs(res.final_elbow_offset()) < 1e-4
