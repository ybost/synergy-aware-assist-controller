"""Minimum-jerk reference (spec section 3)."""

import numpy as np
import pytest

from synergy_aware.arm import fk, ik
from synergy_aware.trajectory import ReachReference, min_jerk_profile

T = 1.0


def test_profile_boundary_conditions():
    s0, v0, a0 = min_jerk_profile(0.0, T)
    s1, v1, a1 = min_jerk_profile(T, T)
    assert (s0, v0, a0) == (0.0, 0.0, 0.0)
    assert (s1, v1, a1) == (1.0, 0.0, 0.0)


def test_profile_midpoint_and_peak_velocity():
    s, v, _ = min_jerk_profile(T / 2, T)
    assert s == pytest.approx(0.5)
    assert v == pytest.approx(1.875 / T)  # known peak of the minimum-jerk profile


def test_profile_derivatives_consistent():
    eps = 1e-6
    for t in (0.1, 0.3, 0.5, 0.8):
        s_p, v_p, _ = min_jerk_profile(t + eps, T)
        s_m, v_m, _ = min_jerk_profile(t - eps, T)
        _, v, a = min_jerk_profile(t, T)
        assert v == pytest.approx((s_p - s_m) / (2 * eps), abs=1e-6)
        assert a == pytest.approx((v_p - v_m) / (2 * eps), abs=1e-6)


def test_profile_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        min_jerk_profile(0.5, 0.0)


def test_reference_starts_and_ends_at_rest_on_targets(params, reach_endpoints):
    x0, x1 = reach_endpoints
    ref = ReachReference(x0, x1, T, params)
    start, end = ref.at(0.0), ref.at(T)
    np.testing.assert_allclose(start.q, ik(x0, params), atol=1e-12)
    np.testing.assert_allclose(end.q, ik(x1, params), atol=1e-12)
    for r in (start, end):
        np.testing.assert_allclose(r.qd, 0.0, atol=1e-12)
        np.testing.assert_allclose(r.qdd, 0.0, atol=1e-12)


def test_reference_holds_outside_the_reach(params, reach_endpoints):
    x0, x1 = reach_endpoints
    ref = ReachReference(x0, x1, T, params)
    np.testing.assert_allclose(ref.at(-0.5).q, ref.at(0.0).q)
    np.testing.assert_allclose(ref.at(T + 2.0).q, ref.at(T).q)
    np.testing.assert_allclose(ref.at(T + 2.0).qd, 0.0, atol=1e-12)


def test_hand_path_is_straight_and_matches_fk(params, reach_endpoints):
    x0, x1 = reach_endpoints
    ref = ReachReference(x0, x1, T, params)
    direction = (x1 - x0) / np.linalg.norm(x1 - x0)
    for t in np.linspace(0.0, T, 11):
        r = ref.at(t)
        np.testing.assert_allclose(fk(r.q, params), r.x, atol=1e-10)
        offset = r.x - x0
        cross = offset[0] * direction[1] - offset[1] * direction[0]
        assert abs(cross) < 1e-10


def test_joint_derivatives_consistent_with_finite_differences(params, reach_endpoints):
    x0, x1 = reach_endpoints
    ref = ReachReference(x0, x1, T, params)
    eps = 1e-6
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        r, rp, rm = ref.at(t), ref.at(t + eps), ref.at(t - eps)
        np.testing.assert_allclose(r.qd, (rp.q - rm.q) / (2 * eps), atol=1e-6)
        np.testing.assert_allclose(r.qdd, (rp.qd - rm.qd) / (2 * eps), atol=1e-5)


def test_unreachable_target_raises(params):
    with pytest.raises(ValueError):
        ReachReference(np.array([0.35, 0.15]), np.array([2.0, 0.0]), T, params).at(T)
