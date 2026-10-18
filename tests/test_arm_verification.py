"""Level 1 numerical verification (spec section 9)."""

import numpy as np
import pytest

from synergy_aware.arm import (
    coriolis_vector,
    forward_dynamics,
    fk,
    ik,
    integrate,
    kinetic_energy,
    mass_matrix,
)


def test_mass_matrix_symmetric_positive_definite(params, q_grid):
    for q in q_grid:
        M = mass_matrix(q, params)
        assert M.shape == (2, 2)
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_zero_input_at_rest_stays_at_rest(params, q_grid):
    for q in q_grid:
        qdd = forward_dynamics(q, np.zeros(2), np.zeros(2), params)
        np.testing.assert_allclose(qdd, 0.0, atol=1e-12)


def test_open_loop_torque_at_rest_gives_Minv_tau(params, q_grid):
    tau = np.array([0.7, -0.3])
    for q in q_grid:
        qdd = forward_dynamics(q, np.zeros(2), tau, params)
        np.testing.assert_allclose(qdd, np.linalg.solve(mass_matrix(q, params), tau), atol=1e-10)


def test_coriolis_vector_matches_euler_lagrange_identity(params, q_grid):
    """Validate the vector C(q,qd) qd, not a particular matrix form of C.

    For T = 0.5 qd^T M(q) qd:  h_i = sum_jk dM_ij/dq_k qd_k qd_j - 0.5 sum_jk dM_jk/dq_i qd_j qd_k.
    """
    rng = np.random.default_rng(0)
    eps = 1e-6
    for q in q_grid:
        qd = rng.uniform(-2.0, 2.0, size=2)
        dM = np.zeros((2, 2, 2))  # dM[k] = dM/dq_k
        for k in range(2):
            dq = np.zeros(2)
            dq[k] = eps
            dM[k] = (mass_matrix(q + dq, params) - mass_matrix(q - dq, params)) / (2 * eps)
        h = np.zeros(2)
        for i in range(2):
            h[i] = sum(dM[k][i, j] * qd[k] * qd[j] for j in range(2) for k in range(2)) - 0.5 * sum(
                dM[i][j, k] * qd[j] * qd[k] for j in range(2) for k in range(2)
            )
        np.testing.assert_allclose(coriolis_vector(q, qd, params), h, atol=1e-5)


def _free_motion(p, dt, T=1.0):
    q0, qd0 = np.array([0.2, 0.8]), np.array([1.0, -0.6])
    return integrate(q0, qd0, lambda t, q, qd: np.zeros(2), p, dt, T)


def test_energy_conserved_without_damping_or_torque(params):
    _, q, qd = _free_motion(params, dt=1e-3)
    ke = np.array([kinetic_energy(qi, qdi, params) for qi, qdi in zip(q, qd)])
    assert abs(ke[-1] - ke[0]) / ke[0] < 1e-6


def test_energy_non_increasing_with_damping(damped_params):
    _, q, qd = _free_motion(damped_params, dt=1e-3)
    ke = np.array([kinetic_energy(qi, qdi, damped_params) for qi, qdi in zip(q, qd)])
    assert np.all(np.diff(ke) <= 1e-9)
    assert ke[-1] < ke[0]


def test_rk4_error_scales_with_timestep(params):
    """Halving dt should cut the error by roughly 2^4 = 16 for RK4; require > 8."""
    tau = lambda t, q, qd: np.array([0.5, 0.2])  # noqa: E731
    q0, qd0 = np.array([0.1, 0.6]), np.array([0.5, 0.0])

    def final_q(dt):
        return integrate(q0, qd0, tau, params, dt, 0.5)[1][-1]

    ref = final_q(1e-3)
    err_coarse = np.linalg.norm(final_q(2e-2) - ref)
    err_fine = np.linalg.norm(final_q(1e-2) - ref)
    assert err_fine > 0
    assert err_coarse / err_fine > 8


def test_fk_known_postures(params):
    L = params.l1 + params.l2
    np.testing.assert_allclose(fk(np.array([0.0, 0.0]), params), [L, 0.0], atol=1e-12)
    np.testing.assert_allclose(fk(np.array([np.pi / 2, 0.0]), params), [0.0, L], atol=1e-12)


def test_ik_fk_round_trip(params, q_grid):
    for q in q_grid:
        np.testing.assert_allclose(ik(fk(q, params), params), q, atol=1e-9)


@pytest.mark.skip(reason="Joint-limit behaviour not yet specified (spec section 2, TBD-CAL)")
def test_joint_limit_behaviour():
    raise AssertionError
