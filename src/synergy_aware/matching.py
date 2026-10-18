"""Matched-budget gain rule for the generic comparator C2 (spec section 5).

C2 adds extra elbow stiffness dK (with damping dD = 2 zeta sqrt(dK I_e)). Its robot effort
J_tau rises with dK but saturates at the cost of fully cancelling the synergy, and the
loop becomes unstable at very high gain. So matching C3's effort is not always possible:
fit_boost_gain reports matched=False, with the highest stable gain, when no stable gain
reaches the target. That outcome is a finding, not an error.

Calibration discipline: fit_boost_rule must be called with calibration-set conditions
only. The resulting BoostGainRule is immutable and is applied unchanged to the final set.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .arm import ArmParams
from .controllers import Controller, ElbowBoostController
from .metrics import robot_effort
from .simulation import TrialConfig, run_trial

STABLE_TORQUE_FRACTION = 0.95  # a trial is "stable" if elbow torque stays below this share of its limit


def elbow_inertia(arm: ArmParams) -> float:
    """Effective inertia seen by the elbow, I2 + m2 lc2^2."""
    return arm.I2 + arm.m2 * arm.lc2**2


def boost_damping(dK: float, arm: ArmParams, zeta: float = 0.7) -> float:
    return 2.0 * zeta * float(np.sqrt(dK * elbow_inertia(arm)))


def boost_controller(base: Controller, dK: float, arm: ArmParams, zeta: float = 0.7) -> ElbowBoostController:
    return ElbowBoostController(base, dK, boost_damping(dK, arm, zeta))


@dataclass(frozen=True)
class BoostFit:
    dK: float  # chosen extra elbow stiffness (N m/rad)
    effort: float  # J_tau achieved by C2 at dK
    target: float  # J_tau of C3 on the same condition
    matched: bool  # False if no stable gain reaches the target


def _evaluate(cfg: TrialConfig, base: Controller, dK: float, zeta: float) -> tuple[float, bool]:
    res = run_trial(cfg, boost_controller(base, dK, cfg.arm, zeta))
    stable = bool(
        np.all(np.isfinite(res.q))
        and np.max(np.abs(res.tau_robot[:, 1])) < STABLE_TORQUE_FRACTION * cfg.limits.tau_max[1]
    )
    return robot_effort(res, cfg.dt), stable


def fit_boost_gain(
    cfg: TrialConfig,
    base: Controller,
    target_effort: float,
    zeta: float = 0.7,
    dK_min: float = 1.0,
    dK_max: float = 1e5,
    grid_factor: float = 2.0,
    rel_tol: float = 0.01,
) -> BoostFit:
    """Find the extra elbow stiffness at which C2's J_tau equals target_effort.

    Scans a geometric grid upward until the loop is no longer stable or dK_max is reached,
    then bisects (in log gain) inside the bracketing interval.
    """
    stable_points: list[tuple[float, float]] = []  # (dK, effort)
    dK = dK_min
    while dK <= dK_max:
        effort, stable = _evaluate(cfg, base, dK, zeta)
        if not stable:
            break
        stable_points.append((dK, effort))
        if effort >= target_effort:
            break
        dK *= grid_factor
    if not stable_points:
        return BoostFit(dK=0.0, effort=float("nan"), target=target_effort, matched=False)
    last_dK, last_effort = stable_points[-1]
    if last_effort < target_effort:
        return BoostFit(dK=last_dK, effort=last_effort, target=target_effort, matched=False)
    if len(stable_points) == 1:  # even the smallest gain already meets or exceeds the target
        return BoostFit(dK=last_dK, effort=last_effort, target=target_effort, matched=abs(last_effort - target_effort) <= rel_tol * target_effort)
    lo, hi = stable_points[-2][0], last_dK
    best = (last_dK, last_effort)
    for _ in range(30):
        mid = float(np.sqrt(lo * hi))
        effort, stable = _evaluate(cfg, base, mid, zeta)
        if not stable:
            hi = mid
            continue
        best = (mid, effort)
        if abs(effort - target_effort) <= rel_tol * target_effort:
            break
        if effort < target_effort:
            lo = mid
        else:
            hi = mid
    matched = abs(best[1] - target_effort) <= rel_tol * target_effort
    return BoostFit(dK=best[0], effort=best[1], target=target_effort, matched=matched)


@dataclass(frozen=True)
class BoostGainRule:
    """Frozen map from cancellation demand x = alpha * u_abd to extra elbow stiffness.

    Log-linear interpolation between calibration points; no extrapolation.
    """

    x: tuple[float, ...]
    dK: tuple[float, ...]
    matched: tuple[bool, ...]

    def __post_init__(self):
        if len(self.x) != len(self.dK) or len(self.x) < 1:
            raise ValueError("x and dK must be non-empty and of equal length")
        if any(b <= a for a, b in zip(self.x, self.x[1:])):
            raise ValueError("x must be strictly increasing")

    def __call__(self, alpha: float, u_abd: float) -> float:
        x = alpha * u_abd
        if x < self.x[0] - 1e-12 or x > self.x[-1] + 1e-12:
            raise ValueError(f"demand {x:.4g} is outside the calibrated range [{self.x[0]:.4g}, {self.x[-1]:.4g}]")
        if len(self.x) == 1:
            return self.dK[0]
        return float(np.exp(np.interp(x, self.x, np.log(self.dK))))


def fit_boost_rule(
    calibration_cfgs: Sequence[TrialConfig],
    c3_factory: Callable[[], Controller],
    base: Controller,
    zeta: float = 0.7,
    **fit_kwargs,
) -> BoostGainRule:
    """Fit the C2 gain rule on CALIBRATION conditions only.

    For each condition, the target is J_tau of a fresh C3 on that condition. Conditions
    where no stable gain reaches the target are kept with matched=False (using the highest
    stable gain), so the report can say C2 could not be budget-matched there.
    """
    rows = []
    for cfg in calibration_cfgs:
        target = robot_effort(run_trial(cfg, c3_factory()), cfg.dt)
        fit = fit_boost_gain(replace(cfg), base, target, zeta=zeta, **fit_kwargs)
        rows.append((cfg.alpha * cfg.u_abd, fit.dK, fit.matched))
    rows.sort()
    return BoostGainRule(x=tuple(r[0] for r in rows), dK=tuple(r[1] for r in rows), matched=tuple(r[2] for r in rows))
