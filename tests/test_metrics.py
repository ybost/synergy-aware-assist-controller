"""Metrics (spec section 7)."""

from dataclasses import replace

import numpy as np
import pytest

from synergy_aware.controllers import ImpedanceController, NullController
from synergy_aware.metrics import (
    ReferenceSet,
    dimensionless_jerk,
    excess_elbow_flexion,
    peak_robot_torque,
    peak_torque_rate,
    robot_effort,
    time_to_target,
)
from synergy_aware.simulation import TrialResult, run_trial
from synergy_aware.trajectory import ReachReference


def _synthetic(params, q_e, tau_robot=None):
    n = len(q_e) - 1
    t = np.linspace(0.0, 1.0, n + 1)
    q = np.column_stack([np.zeros(n + 1), q_e])
    z = np.zeros((n, 2))
    return TrialResult(
        t=t, q=q, qd=np.zeros((n + 1, 2)), q_ref=q.copy(), tau_vol=z, tau_syn=z,
        tau_robot=z if tau_robot is None else tau_robot, x1=np.zeros(2), arm=params,
    )


# --- E_AFE ---------------------------------------------------------------------------------


def test_afe_is_zero_when_elbow_matches_expected(params):
    qe = np.linspace(0.5, 1.0, 101)
    assert excess_elbow_flexion(_synthetic(params, qe), qe) == 0.0


def test_afe_is_one_sided(params):
    qe = np.full(101, 0.8)
    assert excess_elbow_flexion(_synthetic(params, qe), qe + 0.2) == 0.0  # less flexed than expected


def test_afe_constant_excess_equals_delta_squared(params):
    qe = np.full(101, 1.0)
    assert excess_elbow_flexion(_synthetic(params, qe), qe - 0.1) == pytest.approx(0.1**2)


def test_afe_rejects_misaligned_reference(params):
    qe = np.full(101, 1.0)
    with pytest.raises(ValueError):
        excess_elbow_flexion(_synthetic(params, qe), np.full(50, 1.0))


# --- J_tau, timing, constraint metrics -------------------------------------------------------


def test_robot_effort_constant_torque(params):
    n, dt = 100, 0.01
    res = _synthetic(params, np.full(n + 1, 1.0), tau_robot=np.tile([3.0, 4.0], (n, 1)))
    assert robot_effort(res, dt) == pytest.approx(25.0 * n * dt)


def test_peak_torque_and_rate(params):
    tau = np.zeros((50, 2))
    tau[10:, 1] = 6.0
    res = _synthetic(params, np.full(51, 1.0), tau_robot=tau)
    assert peak_robot_torque(res) == 6.0
    assert peak_torque_rate(res, dt=0.001) == pytest.approx(6000.0)


def test_time_to_target_and_unreachable_tolerance(make_cfg):
    res = run_trial(make_cfg(), NullController())
    t_hit = time_to_target(res, tol=2e-3)
    assert 0.5 < t_hit < 2.0
    assert np.isnan(time_to_target(res, tol=1e-12))


# --- Dimensionless jerk ----------------------------------------------------------------------


def _min_jerk_hand_path(params, endpoints, T=1.0, dt=1e-3):
    ref = ReachReference(*endpoints, T, params)
    return np.array([ref.at(t).x for t in np.arange(0.0, T + dt / 2, dt)])


def test_minimum_jerk_reach_has_dimensionless_jerk_720_unfiltered(params, reach_endpoints):
    x = _min_jerk_hand_path(params, reach_endpoints)
    assert dimensionless_jerk(x, 1e-3, cutoff_hz=None) == pytest.approx(720.0, rel=0.02)


def test_dimensionless_jerk_is_insensitive_to_duration_and_amplitude(params, reach_endpoints):
    slow = dimensionless_jerk(_min_jerk_hand_path(params, reach_endpoints, T=2.0), 1e-3, cutoff_hz=None)
    near = (reach_endpoints[0], reach_endpoints[0] + 0.5 * (reach_endpoints[1] - reach_endpoints[0]))
    short = dimensionless_jerk(_min_jerk_hand_path(params, near), 1e-3, cutoff_hz=None)
    assert slow == pytest.approx(720.0, rel=0.02)
    assert short == pytest.approx(720.0, rel=0.02)


def test_dimensionless_jerk_filtered_stays_close_to_720(params, reach_endpoints):
    x = _min_jerk_hand_path(params, reach_endpoints)
    assert dimensionless_jerk(x, 1e-3, cutoff_hz=10.0) == pytest.approx(720.0, rel=0.05)


def test_jerky_movement_scores_higher_than_smooth(params, reach_endpoints):
    x = _min_jerk_hand_path(params, reach_endpoints)
    wiggle = x + 0.002 * np.sin(2 * np.pi * 6.0 * np.linspace(0, 1, len(x)))[:, None]
    assert dimensionless_jerk(wiggle, 1e-3, cutoff_hz=None) > 2 * dimensionless_jerk(x, 1e-3, cutoff_hz=None)


# --- Frozen reference and the C0 vs C1 comparison ----------------------------------------------


def test_reference_run_scores_zero_against_itself(make_cfg, impedance_gains):
    refs = ReferenceSet(ImpedanceController(impedance_gains))
    cfg = make_cfg(t_end=1.5)
    res = run_trial(replace(cfg, alpha=0.0), ImpedanceController(impedance_gains))
    assert excess_elbow_flexion(res, refs.get(cfg)) == 0.0


def test_reference_is_shared_across_alpha_and_u_abd_and_cached(make_cfg, impedance_gains):
    refs = ReferenceSet(ImpedanceController(impedance_gains))
    a = refs.get(make_cfg(t_end=1.0, alpha=0.0, u_abd=0.0))
    b = refs.get(make_cfg(t_end=1.0, alpha=0.5, u_abd=1.0))
    assert a is b


def test_frozen_reference_set_rejects_new_conditions(make_cfg, impedance_gains):
    refs = ReferenceSet(ImpedanceController(impedance_gains))
    refs.get(make_cfg(t_end=1.0))
    refs.freeze()
    refs.get(make_cfg(t_end=1.0, alpha=0.3))  # already generated, still fine
    with pytest.raises(KeyError):
        refs.get(make_cfg(t_end=1.5))


def test_c1_reduces_excess_flexion_relative_to_c0(make_cfg, impedance_gains):
    """The C0 vs C1 gap the spec says must be reported (section 4.2 note)."""
    refs = ReferenceSet(ImpedanceController(impedance_gains))
    cfg = make_cfg(alpha=0.2, u_abd=1.0)
    ref_qe = refs.get(cfg)
    e_c0 = excess_elbow_flexion(run_trial(cfg, NullController()), ref_qe)
    e_c1 = excess_elbow_flexion(run_trial(cfg, ImpedanceController(impedance_gains)), ref_qe)
    assert e_c1 > 0.0
    assert e_c1 < e_c0
