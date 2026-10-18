"""The estimator must not have access to simulator ground truth (spec section 6, charter 19.2)."""

import dataclasses
import inspect

import pytest

from synergy_aware.estimator import Observation, SynergyEstimator

ALLOWED_FIELDS = {"t", "q", "qd", "q_ref", "qd_ref", "u_abd", "tau_robot_prev"}
FORBIDDEN_NAMES = {"tau_syn", "alpha", "alpha_true", "ground_truth", "q_e_expected_offline"}


def test_observation_fields_are_exactly_the_allowed_set():
    names = {f.name for f in dataclasses.fields(Observation)}
    assert not names & FORBIDDEN_NAMES
    assert names == ALLOWED_FIELDS


def test_observation_is_immutable():
    assert Observation.__dataclass_params__.frozen


def test_update_accepts_only_an_observation():
    assert list(inspect.signature(SynergyEstimator.update).parameters) == ["self", "obs"]


def test_constructor_takes_no_ground_truth_arguments():
    params = set(inspect.signature(SynergyEstimator.__init__).parameters) - {"self"}
    assert not params & FORBIDDEN_NAMES


@pytest.mark.skip(reason="Needs the simulator loop; add a leak test that varies true alpha with identical observations")
def test_estimate_independent_of_true_alpha_given_same_observations():
    raise AssertionError
