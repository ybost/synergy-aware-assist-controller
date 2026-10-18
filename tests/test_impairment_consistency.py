"""Level 2 internal model consistency (spec section 9).

Passing these verifies the implementation matches the specification. It does NOT
establish physiological validity (charter section 18).
"""

import numpy as np
import pytest

from synergy_aware.impairment import synergy_torque

TAU_SCALE = 10.0  # test-only value


def test_alpha_zero_gives_no_impairment_torque():
    for u in (0.0, 0.5, 1.0):
        np.testing.assert_array_equal(synergy_torque(0.0, u, TAU_SCALE), [0.0, 0.0])


def test_u_abd_zero_gives_no_impairment_torque():
    for alpha in (0.0, 0.3, 1.0):
        np.testing.assert_array_equal(synergy_torque(alpha, 0.0, TAU_SCALE), [0.0, 0.0])


def test_matches_closed_form():
    np.testing.assert_allclose(synergy_torque(0.4, 0.5, TAU_SCALE), [0.0, 0.4 * 0.5 * TAU_SCALE])


def test_monotonic_in_alpha():
    vals = [synergy_torque(a, 0.7, TAU_SCALE)[1] for a in np.linspace(0, 1, 11)]
    assert np.all(np.diff(vals) > 0)


def test_monotonic_in_u_abd():
    vals = [synergy_torque(0.5, u, TAU_SCALE)[1] for u in np.linspace(0, 1, 11)]
    assert np.all(np.diff(vals) > 0)


def test_acts_only_on_elbow_in_flexion_direction():
    tau = synergy_torque(0.5, 0.8, TAU_SCALE)
    assert tau[0] == 0.0  # shoulder channel untouched
    assert tau[1] > 0.0  # flexion is positive (spec section 1)


@pytest.mark.parametrize("alpha,u,scale", [(-0.1, 0.5, 10.0), (0.5, -0.1, 10.0), (0.5, 1.1, 10.0), (0.5, 0.5, 0.0)])
def test_rejects_out_of_range_inputs(alpha, u, scale):
    with pytest.raises(ValueError):
        synergy_torque(alpha, u, scale)
