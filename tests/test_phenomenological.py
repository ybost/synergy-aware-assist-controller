"""Level 3 and 4 placeholders (spec section 9). Skipped until the human model and controllers exist.

Level 3 runs under C0, and separately under C1; never under C3, whose purpose is to
counteract the effect being validated.
"""

import pytest

NEEDS = "needs human model and controllers (spec sections 4.1 and 5)"


@pytest.mark.skip(reason=NEEDS)
def test_reachable_workspace_decreases_monotonically_with_u_abd():
    raise AssertionError


@pytest.mark.skip(reason=NEEDS)
def test_elbow_flexion_excess_rises_with_u_abd_and_alpha():
    raise AssertionError


@pytest.mark.skip(reason=NEEDS)
def test_robust_to_altered_arm_parameters():
    raise AssertionError


@pytest.mark.skip(reason=NEEDS)
def test_robust_to_unseen_alpha_and_directions():
    raise AssertionError


@pytest.mark.skip(reason=NEEDS)
def test_robust_to_sensor_noise():
    raise AssertionError
