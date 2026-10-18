"""C2 (generic elbow boost) and its matched-budget gain rule (spec section 5)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from synergy_aware.controllers import ElbowBoostController, ImpedanceController, NullController
from synergy_aware.estimator import Observation, SynergyEstimator
from synergy_aware.matching import (
    BoostGainRule,
    boost_controller,
    boost_damping,
    elbow_inertia,
    fit_boost_gain,
    fit_boost_rule,
)
from synergy_aware.metrics import robot_effort
from synergy_aware.simulation import run_trial


def _obs(e=0.0, ed=0.0, u=1.0):
    return Observation(
        t=0.0,
        q=np.array([0.3, 1.0 + e]),
        qd=np.array([0.0, ed]),
        q_ref=np.array([0.3, 1.0]),
        qd_ref=np.zeros(2),
        u_abd=u,
        tau_robot_prev=np.zeros(2),
    )


def test_boost_adds_elbow_pd_only(impedance_gains):
    base = ImpedanceController(impedance_gains)
    c2 = ElbowBoostController(base, dK=300.0, dD=7.0)
    obs = _obs(e=0.02, ed=0.1)
    extra = c2.torque(obs) - base.torque(obs)
    np.testing.assert_allclose(extra, [0.0, 300.0 * (-0.02) + 7.0 * (-0.1)])


def test_boost_is_generic_and_ignores_u_abd(impedance_gains):
    c2 = ElbowBoostController(ImpedanceController(impedance_gains), 300.0, 7.0)
    np.testing.assert_array_equal(c2.torque(_obs(e=0.02, u=0.0)), c2.torque(_obs(e=0.02, u=1.0)))


def test_boost_is_negligible_without_a_synergy(make_cfg, impedance_gains):
    """Plain C1 already tracks to ~4 microradians (zero-order-hold error); the boost only shrinks that."""
    cfg = make_cfg(t_end=1.5)
    base = ImpedanceController(impedance_gains)
    plain = run_trial(cfg, base)
    boosted = run_trial(cfg, boost_controller(base, 1000.0, cfg.arm))
    np.testing.assert_allclose(boosted.q, plain.q, atol=1e-4)


def test_boost_validation(impedance_gains):
    with pytest.raises(ValueError):
        ElbowBoostController(ImpedanceController(impedance_gains), -1.0, 1.0)


def test_static_offset_follows_the_stiffness_balance(make_cfg, impedance_gains, human_params):
    cfg = make_cfg(alpha=0.04, u_abd=1.0, t_end=3.0)
    dK = 500.0
    res = run_trial(cfg, boost_controller(ImpedanceController(impedance_gains), dK, cfg.arm))
    tau_syn_e = 0.04 * human_params.tau_max_neg[1]
    expected = tau_syn_e / (human_params.K[1] + impedance_gains.Kp[1] + dK)
    assert res.final_elbow_offset() == pytest.approx(expected, rel=0.03)


def test_effort_rises_with_gain_but_saturates(make_cfg, impedance_gains):
    cfg = make_cfg(alpha=0.2, u_abd=1.0, t_end=2.0, dt=2e-3)
    base = ImpedanceController(impedance_gains)
    efforts = [robot_effort(run_trial(cfg, boost_controller(base, dK, cfg.arm)), cfg.dt) for dK in (0.0, 100.0, 1000.0, 3000.0)]
    assert np.all(np.diff(efforts) > 0)
    full_cancel = (0.2 * 25.0) ** 2 * cfg.t_end  # cost of fully cancelling the synergy
    assert efforts[-1] < full_cancel


def test_damping_rule():
    from synergy_aware.arm import ArmParams

    arm = ArmParams(0.3, 0.35, 2.0, 1.5, 0.13, 0.17, 0.02, 0.03)
    assert elbow_inertia(arm) == pytest.approx(0.03 + 1.5 * 0.17**2)
    assert boost_damping(400.0, arm, zeta=0.5) == pytest.approx(2 * 0.5 * np.sqrt(400.0 * elbow_inertia(arm)))


def test_fit_recovers_a_known_gain(make_cfg, impedance_gains):
    cfg = make_cfg(alpha=0.2, u_abd=1.0, t_end=1.5, dt=2e-3)
    base = ImpedanceController(impedance_gains)
    target = robot_effort(run_trial(cfg, boost_controller(base, 300.0, cfg.arm)), cfg.dt)
    fit = fit_boost_gain(cfg, base, target)
    assert fit.matched
    assert fit.effort == pytest.approx(target, rel=0.01)
    assert fit.dK == pytest.approx(300.0, rel=0.15)


def test_unreachable_budget_is_reported_not_forced(make_cfg, impedance_gains):
    """Effort saturates at the full-cancellation cost, so a larger target cannot be matched."""
    cfg = make_cfg(alpha=0.2, u_abd=1.0, t_end=1.5, dt=2e-3)
    fit = fit_boost_gain(cfg, ImpedanceController(impedance_gains), target_effort=1e6)
    assert fit.matched is False
    assert fit.dK > 0.0 and np.isfinite(fit.effort) and fit.effort < 1e6


def test_rule_interpolates_log_linearly_and_refuses_to_extrapolate():
    rule = BoostGainRule(x=(0.1, 0.2), dK=(100.0, 10000.0), matched=(True, True))
    assert rule(0.1, 1.0) == pytest.approx(100.0)
    assert rule(0.15, 1.0) == pytest.approx(1000.0)  # geometric midpoint
    assert rule(0.3, 0.5) == pytest.approx(1000.0)  # depends on alpha * u_abd only
    with pytest.raises(ValueError):
        rule(0.05, 1.0)
    with pytest.raises(ValueError):
        rule(0.5, 1.0)


def test_rule_is_immutable_and_validated():
    rule = BoostGainRule(x=(0.1, 0.2), dK=(100.0, 200.0), matched=(True, True))
    with pytest.raises(FrozenInstanceError):
        rule.dK = (1.0, 2.0)
    with pytest.raises(ValueError):
        BoostGainRule(x=(0.2, 0.1), dK=(1.0, 2.0), matched=(True, True))


@pytest.mark.slow
def test_fit_rule_end_to_end_on_calibration_conditions(make_cfg, impedance_gains, human_params):
    base = ImpedanceController(impedance_gains)
    ts = human_params.tau_max_neg[1]

    def c3():
        from synergy_aware.controllers import SynergyAwareController

        return SynergyAwareController(base, SynergyEstimator(ts, human_params.K[1], 50, 0.05), ts)

    cals = [make_cfg(alpha=a, u_abd=1.0, t_end=1.5, dt=2e-3) for a in (0.1, 0.2)]
    rule = fit_boost_rule(cals, c3, base)
    assert rule.x == pytest.approx((0.1, 0.2))
    assert all(rule.matched)
    # In the linear regime both C2's and C3's effort scale with alpha^2, so the matching
    # gain is the same at every severity; it can only depend on severity through
    # saturation or estimator transients. Document that, rather than assume otherwise.
    assert rule.dK[0] > 0.0
    assert rule.dK[1] == pytest.approx(rule.dK[0], rel=0.05)
