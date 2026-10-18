"""Evaluation metrics (spec section 7, charter section 16).

Elbow convention: flexion positive. All integrals use the trial's own sample times.
"""

from dataclasses import astuple, replace

import numpy as np
from scipy.signal import butter, filtfilt

from .arm import fk
from .controllers import Controller
from .simulation import TrialConfig, TrialResult, run_trial


def hand_positions(result: TrialResult) -> np.ndarray:
    return np.array([fk(q, result.arm) for q in result.q])


def excess_elbow_flexion(result: TrialResult, q_e_expected: np.ndarray) -> float:
    """E_AFE = (1/T) * integral of max(0, q_e - q_e_expected)^2 dt (flexion excess only)."""
    if q_e_expected.shape != result.t.shape:
        raise ValueError("q_e_expected must be sampled on the trial's time grid")
    excess = np.maximum(0.0, result.q[:, 1] - q_e_expected)
    duration = result.t[-1] - result.t[0]
    return float(np.trapezoid(excess**2, result.t) / duration)


def robot_effort(result: TrialResult, dt: float) -> float:
    """J_tau = integral of ||tau_robot||^2 dt. Torques are held constant over each dt."""
    return float(np.sum(np.sum(result.tau_robot**2, axis=1)) * dt)


def time_to_target(result: TrialResult, tol: float) -> float:
    """First time after which the hand stays within tol of the target; NaN if it never does."""
    within = np.linalg.norm(hand_positions(result) - result.x1, axis=1) <= tol
    stays = np.logical_and.accumulate(within[::-1])[::-1]
    return float(result.t[np.argmax(stays)]) if stays.any() else float("nan")


def peak_robot_torque(result: TrialResult) -> float:
    return float(np.max(np.abs(result.tau_robot)))


def peak_torque_rate(result: TrialResult, dt: float) -> float:
    steps = np.abs(np.diff(result.tau_robot, axis=0, prepend=np.zeros((1, 2))))
    return float(np.max(steps) / dt)


def dimensionless_jerk(x: np.ndarray, dt: float, cutoff_hz: float | None = 10.0) -> float:
    """Dimensionless squared jerk of a sampled hand path, x of shape (n, 2).

    DJ = (T^5 / L^2) * integral of |jerk|^2 dt, with T the window length and L the path
    length (Hogan & Sternad, 2009; verify citation details before use). A minimum-jerk
    reach gives DJ = 720. Method: optional zero-phase Butterworth low-pass (order 4,
    applied forward and backward), then central differences for velocity, acceleration
    and jerk. Pass the movement window only, not the hold at the target.
    """
    if cutoff_hz is not None:
        b, a = butter(4, cutoff_hz, btype="low", fs=1.0 / dt)
        x = filtfilt(b, a, x, axis=0)
    vel = np.gradient(x, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    T = dt * (len(x) - 1)
    L = float(np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1)))
    return float(T**5 / L**2 * np.trapezoid(np.sum(jerk**2, axis=1), dx=dt))


def _key(cfg: TrialConfig) -> tuple:
    return (
        astuple(cfg.arm),
        tuple(cfg.nominal and astuple(cfg.nominal) or ()),
        tuple(np.asarray(cfg.human.K).tolist() + np.asarray(cfg.human.D).tolist()),
        tuple(np.asarray(cfg.human.tau_max_pos).tolist() + np.asarray(cfg.human.tau_max_neg).tolist()),
        tuple(np.asarray(cfg.limits.tau_max).tolist() + np.asarray(cfg.limits.rate_max).tolist()),
        tuple(np.asarray(cfg.x0).tolist() + np.asarray(cfg.x1).tolist()),
        cfg.T,
        cfg.t_end,
        cfg.dt,
    )


class ReferenceSet:
    """Frozen metric reference: the elbow trajectory with alpha = 0 under the C1 controller.

    alpha = 0 removes the synergy, and C1 does not use u_abd, so the reference does not
    depend on u_abd; one reference per target and trial setup is shared by all controllers
    and all abduction levels. After freeze(), unseen conditions raise KeyError, so the
    benchmark cannot be extended after C2/C3 have been evaluated (spec decision 4).
    """

    def __init__(self, c1_controller: Controller):
        self._controller = c1_controller
        self._cache: dict[tuple, np.ndarray] = {}
        self._frozen = False

    def get(self, cfg: TrialConfig) -> np.ndarray:
        key = _key(cfg)
        if key not in self._cache:
            if self._frozen:
                raise KeyError("reference set is frozen; this condition was not generated before freezing")
            self._cache[key] = run_trial(replace(cfg, alpha=0.0), self._controller).q[:, 1].copy()
        return self._cache[key]

    def freeze(self) -> None:
        self._frozen = True
