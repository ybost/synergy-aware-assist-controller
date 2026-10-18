"""Closed-loop trial runner.

Separates the simulated person (voluntary command plus impairment) from the robot
controller. The impairment torque is recorded as ground truth in the result but is never
passed to a controller: controllers only receive an Observation.

Torques are held constant over each step of length dt (zero-order hold), so dt is also
the controller sampling period.
"""

from dataclasses import dataclass

import numpy as np

from .arm import ArmParams, fk, integrate
from .controllers import Controller, RobotLimits, limit_torque
from .estimator import Observation
from .human import HumanParams, voluntary_torque
from .impairment import synergy_torque
from .trajectory import ReachReference


@dataclass(frozen=True)
class TrialConfig:
    arm: ArmParams  # the real arm
    human: HumanParams
    limits: RobotLimits
    x0: np.ndarray
    x1: np.ndarray
    T: float  # reach duration (s)
    t_end: float  # total simulated time, including hold at the target (s)
    dt: float
    alpha: float = 0.0
    u_abd: float = 0.0
    nominal: ArmParams | None = None  # the person's internal model; defaults to the real arm


@dataclass(frozen=True)
class TrialResult:
    t: np.ndarray  # (n+1,)
    q: np.ndarray  # (n+1, 2)
    qd: np.ndarray  # (n+1, 2)
    q_ref: np.ndarray  # (n+1, 2)
    tau_vol: np.ndarray  # (n, 2)
    tau_syn: np.ndarray  # (n, 2) ground truth, simulator-only
    tau_robot: np.ndarray  # (n, 2) after limits
    x1: np.ndarray
    arm: ArmParams

    def endpoint_error(self) -> float:
        return float(np.linalg.norm(fk(self.q[-1], self.arm) - self.x1))

    def final_elbow_offset(self) -> float:
        """Elbow angle minus its reference at the end of the trial (flexion positive)."""
        return float(self.q[-1, 1] - self.q_ref[-1, 1])


def run_trial(cfg: TrialConfig, controller: Controller) -> TrialResult:
    reset = getattr(controller, "reset", None)
    if callable(reset):
        reset()  # stateful controllers must not carry state between trials
    nominal = cfg.nominal if cfg.nominal is not None else cfg.arm
    ref = ReachReference(cfg.x0, cfg.x1, cfg.T, cfg.arm)
    tau_syn = synergy_torque(cfg.alpha, cfg.u_abd, cfg.human.tau_max_neg[1])  # tau_scale = tau_ext,max,e

    n = int(round(cfg.t_end / cfg.dt))
    t = np.arange(n + 1) * cfg.dt
    q = np.empty((n + 1, 2))
    qd = np.zeros((n + 1, 2))
    q_ref = np.empty((n + 1, 2))
    tau_vol = np.empty((n, 2))
    tau_robot = np.empty((n, 2))

    r0 = ref.at(0.0)
    q[0], q_ref[0] = r0.q, r0.q
    tau_prev = np.zeros(2)

    for k in range(n):
        r = ref.at(t[k])
        obs = Observation(
            t=t[k], q=q[k], qd=qd[k], q_ref=r.q, qd_ref=r.qd, u_abd=cfg.u_abd, tau_robot_prev=tau_prev
        )
        tau_robot[k] = limit_torque(np.asarray(controller.torque(obs), dtype=float), tau_prev, cfg.limits, cfg.dt)
        tau_vol[k] = voluntary_torque(q[k], qd[k], r, cfg.human, nominal)
        total = tau_vol[k] + tau_syn + tau_robot[k]
        _, qs, qds = integrate(q[k], qd[k], lambda _t, _q, _qd, tau=total: tau, cfg.arm, cfg.dt, cfg.dt)
        q[k + 1], qd[k + 1] = qs[-1], qds[-1]
        q_ref[k + 1] = ref.at(t[k + 1]).q
        tau_prev = tau_robot[k]

    return TrialResult(
        t=t,
        q=q,
        qd=qd,
        q_ref=q_ref,
        tau_vol=tau_vol,
        tau_syn=np.tile(tau_syn, (n, 1)),
        tau_robot=tau_robot,
        x1=cfg.x1,
        arm=cfg.arm,
    )
