"""Simulated voluntary motor command (spec section 4.1)."""

import inspect
from dataclasses import replace

import numpy as np
import pytest

from synergy_aware.arm import forward_dynamics, fk, integrate
from synergy_aware.human import HumanParams, feedforward_torque, voluntary_torque
from synergy_aware.impairment import synergy_torque
from synergy_aware.trajectory import ReachReference, Reference

T = 1.0


def test_perfect_tracking_without_saturation_realises_the_reference(params, human_params, reach_endpoints):
    """At q = q_ref the command is pure inverse dynamics, so the arm accelerates as planned."""
    ref = ReachReference(*reach_endpoints, T, params)
    for t in (0.2, 0.5, 0.8):
        r = ref.at(t)
        tau = voluntary_torque(r.q, r.qd, r, human_params, params)
        np.testing.assert_allclose(forward_dynamics(r.q, r.qd, tau, params), r.qdd, atol=1e-9)


def test_feedback_is_restoring(params, human_params):
    at_rest = Reference(x=fk(np.array([0.3, 1.0]), params), q=np.array([0.3, 1.0]), qd=np.zeros(2), qdd=np.zeros(2))
    below = voluntary_torque(at_rest.q - 0.05, np.zeros(2), at_rest, human_params, params)
    above = voluntary_torque(at_rest.q + 0.05, np.zeros(2), at_rest, human_params, params)
    assert np.all(below > 0.0) and np.all(above < 0.0)
    np.testing.assert_allclose(below, human_params.K * 0.05, rtol=1e-9)


def test_saturation_is_direction_specific(params):
    h = HumanParams(
        K=np.array([1000.0, 1000.0]),
        D=np.array([1.0, 1.0]),
        tau_max_pos=np.array([10.0, 12.0]),
        tau_max_neg=np.array([20.0, 8.0]),
    )
    q = np.array([0.3, 1.0])
    ref = Reference(x=fk(q, params), q=q, qd=np.zeros(2), qdd=np.zeros(2))
    up = voluntary_torque(q - 1.0, np.zeros(2), ref, h, params)  # huge positive demand
    down = voluntary_torque(q + 1.0, np.zeros(2), ref, h, params)  # huge negative demand
    np.testing.assert_allclose(up, [10.0, 12.0])
    np.testing.assert_allclose(down, [-20.0, -8.0])


def test_saturation_applies_to_the_total_command(params, human_params, reach_endpoints):
    """Feedforward alone near its limit must not be exceeded once feedback is added."""
    r = ReachReference(*reach_endpoints, T, params).at(0.3)
    squeezed = replace(
        human_params,
        tau_max_pos=np.abs(feedforward_torque(r, params)) + 1e-3,
        tau_max_neg=np.abs(feedforward_torque(r, params)) + 1e-3,
    )
    tau = voluntary_torque(r.q - 0.5, r.qd, r, squeezed, params)
    assert np.all(tau <= squeezed.tau_max_pos + 1e-12)


def test_human_is_unaware_of_the_synergy():
    """Neither the command nor the feedforward can take impairment quantities as input."""
    forbidden = {"tau_syn", "alpha", "u_abd", "tau_scale"}
    for fn in (voluntary_torque, feedforward_torque):
        assert not set(inspect.signature(fn).parameters) & forbidden


def test_params_validation():
    with pytest.raises(ValueError):
        HumanParams(K=np.array([1.0, -1.0]), D=np.ones(2), tau_max_pos=np.ones(2), tau_max_neg=np.ones(2))
    with pytest.raises(ValueError):
        HumanParams(K=np.ones(3), D=np.ones(2), tau_max_pos=np.ones(2), tau_max_neg=np.ones(2))


def _simulate_reach(params, human_params, endpoints, tau_syn, t_end=3.0, dt=1e-3):
    ref = ReachReference(*endpoints, T, params)
    r0 = ref.at(0.0)

    def tau(t, q, qd):
        return voluntary_torque(q, qd, ref.at(t), human_params, params) + tau_syn

    t, q, qd = integrate(r0.q, r0.qd, tau, params, dt, t_end)
    return ref, t, q, qd


def test_unimpaired_human_reaches_the_target(damped_params, human_params, reach_endpoints):
    ref, _, q, _ = _simulate_reach(damped_params, human_params, reach_endpoints, np.zeros(2))
    assert np.linalg.norm(fk(q[-1], damped_params) - reach_endpoints[1]) < 1e-3


def test_constant_elbow_synergy_leaves_a_static_flexion_offset(damped_params, human_params, reach_endpoints):
    """At rest, K_e (q_ref,e - q_e) + tau_syn = 0, so the elbow settles tau_syn / K_e into flexion."""
    tau_syn = synergy_torque(alpha=0.04, u_abd=1.0, tau_scale=human_params.tau_max_neg[1])
    ref, _, q, _ = _simulate_reach(damped_params, human_params, reach_endpoints, tau_syn)
    offset = q[-1][1] - ref.at(T).q[1]
    expected = tau_syn[1] / human_params.K[1]
    assert offset > 0.0  # flexion is positive
    assert offset == pytest.approx(expected, rel=0.02)
