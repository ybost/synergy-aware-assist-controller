"""EXPLORATORY comparison: C0, C1, C2 (budget-matched), C3 and the oracle across noise levels.

Uses PLACEHOLDER parameters (synergy_aware.placeholders), so nothing here is a result.
The point is to exercise the whole pipeline and see whether the figures are sensible.

Protocol notes
- C2's extra elbow gain is fitted per (alpha, noise level) to match C3's robot effort on a
  FIT seed, then evaluated on different EVAL seeds. Fitting never sees the evaluation noise.
- The metric reference is the frozen alpha = 0, noise-free C1 elbow trajectory.
- The oracle uses the true alpha and is an upper bound, not a realistic controller.

Usage:  python scripts/exploratory_comparison.py [--quick]
Outputs: results/exploratory/{trials.csv, summary.csv, fig_*.png}
"""

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from synergy_aware import placeholders as ph
from synergy_aware.controllers import (
    ImpedanceController,
    NullController,
    OracleController,
    SynergyAwareController,
)
from synergy_aware.estimator import SynergyEstimator
from synergy_aware.matching import boost_controller, fit_boost_gain
from synergy_aware.metrics import (
    ReferenceSet,
    dimensionless_jerk,
    excess_elbow_flexion,
    hand_positions,
    peak_robot_torque,
    peak_torque_rate,
    robot_effort,
)
from synergy_aware.simulation import NoiseConfig, run_trial

OUT = Path(__file__).resolve().parents[1] / "results" / "exploratory"
CONTROLLERS = ("C0", "C1", "C2", "C3", "oracle")
ALPHAS = (0.1, 0.2, 0.3)
U_ABD = 1.0
# (label, q_sd [rad], qd_sd [rad/s]); qd_sd is an assumption: 100 x q_sd, i.e. filtered differencing.
NOISE = {
    "none": None,
    "low": (5e-4, 0.05),
    "medium": (1e-3, 0.10),
    "high": (3e-3, 0.30),
}
FIT_SEED = 100
EVAL_SEEDS = (0, 1, 2)
WINDOW = 100  # estimator window (samples)


def noise_cfg(level: str, seed: int):
    spec = NOISE[level]
    return None if spec is None else NoiseConfig(q_sd=spec[0], qd_sd=spec[1], seed=seed)


def make_controllers(alpha: float, dK: float, c1, arm, human):
    ts = human.tau_max_neg[1]
    return {
        "C0": NullController(),
        "C1": c1,
        "C2": boost_controller(c1, dK, arm),
        "C3": SynergyAwareController(c1, SynergyEstimator(ts, human.K[1], WINDOW, 0.05), ts),
        "oracle": OracleController(c1, alpha, ts),
    }


def metrics_row(res, cfg, ref_qe, ctl):
    win = int(round(cfg.T / cfg.dt)) + 1
    tau_e = np.abs(res.tau_robot[:, 1])
    row = {
        "E_AFE": excess_elbow_flexion(res, ref_qe),
        "endpoint_mm": res.endpoint_error() * 1e3,
        "J_tau": robot_effort(res, cfg.dt),
        "peak_tau": peak_robot_torque(res),
        "peak_rate": peak_torque_rate(res, cfg.dt),
        "sat_frac": float(np.mean(tau_e >= 0.999 * cfg.limits.tau_max[1])),
        "DJ_reach": dimensionless_jerk(hand_positions(res)[:win], cfg.dt),
        "alpha_hat_err": float("nan"),
    }
    if isinstance(ctl, SynergyAwareController):
        tail = [e.alpha_hat for e in ctl.estimates[-500:] if e.alpha_hat is not None]
        if tail:
            row["alpha_hat_err"] = float(np.mean(tail) - cfg.alpha)
    return row


def run(quick: bool):
    alphas = (0.2,) if quick else ALPHAS
    levels = ("none", "medium") if quick else tuple(NOISE)
    seeds = (0,) if quick else EVAL_SEEDS
    arm, human = ph.arm(), ph.human()
    c1 = ImpedanceController(ph.c1_gains())
    refs = ReferenceSet(c1)
    ts = human.tau_max_neg[1]
    rows, traces = [], {}

    for alpha in alphas:
        for level in levels:
            fit_cfg = ph.base_cfg(alpha=alpha, u_abd=U_ABD, noise=noise_cfg(level, FIT_SEED))
            c3_effort = robot_effort(
                run_trial(fit_cfg, make_controllers(alpha, 0.0, c1, arm, human)["C3"]), fit_cfg.dt
            )
            fit = fit_boost_gain(fit_cfg, c1, c3_effort, grid_factor=4.0, rel_tol=0.03)
            print(f"alpha={alpha} noise={level:6s} C3 J={c3_effort:7.2f}  C2 dK={fit.dK:8.1f} matched={fit.matched}", flush=True)
            for seed in seeds if level != "none" else (0,):
                cfg = ph.base_cfg(alpha=alpha, u_abd=U_ABD, noise=noise_cfg(level, seed))
                ref_qe = refs.get(cfg)
                for name, ctl in make_controllers(alpha, fit.dK, c1, arm, human).items():
                    res = run_trial(cfg, ctl)
                    row = metrics_row(res, cfg, ref_qe, ctl)
                    row.update(alpha=alpha, noise=level, seed=seed, controller=name, C2_dK=fit.dK, C2_matched=fit.matched)
                    rows.append(row)
                    if alpha == 0.2 and level == "none":
                        traces[name] = (res.t, res.q[:, 1] - ref_qe)
    return rows, traces


KEYS = ["E_AFE", "endpoint_mm", "J_tau", "peak_tau", "peak_rate", "sat_frac", "DJ_reach", "alpha_hat_err"]


def summarise(rows):
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["alpha"], r["noise"], r["controller"]), []).append(r)
    for (alpha, noise, ctl), rs in groups.items():
        d = {"alpha": alpha, "noise": noise, "controller": ctl, "n_seeds": len(rs), "C2_matched": rs[0]["C2_matched"]}
        for k in KEYS:
            vals = np.array([r[k] for r in rs], dtype=float)
            d[k + "_mean"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
            d[k + "_sd"] = float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else float("nan")
        out.append(d)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- figures -------------------------------------------------------------------------------

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
STYLE = {  # fixed identity per controller across all figures
    "C0": dict(color="#898781", ls="-", label="C0 unassisted"),
    "C1": dict(color="#1baf7a", ls="-", label="C1 impedance"),
    "C2": dict(color="#eb6834", ls="-", label="C2 generic boost (matched effort)"),
    "C3": dict(color="#2a78d6", ls="-", label="C3 synergy-aware"),
    "oracle": dict(color="#52514e", ls="--", label="oracle (true alpha; upper bound)"),
}
FOOTER = "EXPLORATORY: placeholder parameters, not calibrated. Dots are individual seeds."


def _axes(plt, title, ylabel):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    fig.text(0.01, 0.01, FOOTER, fontsize=8, color=MUTED)
    return fig, ax


def metric_vs_noise(plt, rows, metric, alpha, title, ylabel, fname, log=True, skip=()):
    levels = [lv for lv in NOISE if any(r["noise"] == lv for r in rows)]
    fig, ax = _axes(plt, title, ylabel)
    for name in [c for c in CONTROLLERS if c not in skip]:
        st = STYLE[name]
        means = []
        for i, lv in enumerate(levels):
            vals = [max(r[metric], 1e-12) for r in rows if r["alpha"] == alpha and r["noise"] == lv and r["controller"] == name]
            ax.plot([i] * len(vals), vals, "o", color=st["color"], ms=5, alpha=0.45, mec="none")
            means.append(np.mean(vals))
        ax.plot(range(len(levels)), means, st["ls"], color=st["color"], lw=2, label=st["label"], marker="o", ms=7, mec=SURFACE, mew=1.5)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{lv}\n(q sd {NOISE[lv][0]*1e3:.1f} mrad)" if NOISE[lv] else "none" for lv in levels], color=INK2)
    ax.set_xlabel("sensor noise level", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)


def trace_figure(plt, traces, fname):
    fig, ax = _axes(plt, "Elbow angle relative to the frozen reference, no sensor noise (alpha = 0.2)", "elbow angle minus reference (rad)")
    for name in CONTROLLERS:
        t, d = traces[name]
        st = STYLE[name]
        ax.plot(t, d, st["ls"], color=st["color"], lw=2, label=st["label"])
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.text(1.02, 0.65, "reach ends", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xlabel("time (s)", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)


def load_rows():
    rows = []
    for r in csv.DictReader(open(OUT / "trials.csv")):
        for k in KEYS + ["alpha", "C2_dK"]:
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
        r["C2_matched"] = r["C2_matched"] == "True"
        rows.append(r)
    return rows


def rebuild_traces(rows):
    """Re-run only the alpha = 0.2, noise-free trials (for the trace figure) from saved C2 gains."""
    dK = next(r["C2_dK"] for r in rows if r["alpha"] == 0.2 and r["noise"] == "none")
    arm, human = ph.arm(), ph.human()
    c1 = ImpedanceController(ph.c1_gains())
    cfg = ph.base_cfg(alpha=0.2, u_abd=U_ABD)
    ref_qe = ReferenceSet(c1).get(cfg)
    return {n: (lambda res: (res.t, res.q[:, 1] - ref_qe))(run_trial(cfg, c)) for n, c in make_controllers(0.2, dK, c1, arm, human).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one severity, two noise levels, one seed")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from the saved trials.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        rows = load_rows()
        traces = rebuild_traces(rows)
    else:
        rows, traces = run(args.quick)
        write_csv(OUT / "trials.csv", rows)
        write_csv(OUT / "summary.csv", summarise(rows))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_vs_noise(plt, rows, "E_AFE", 0.2, "Excess elbow flexion vs sensor noise (alpha = 0.2)", "E_AFE (rad^2, log scale)", "fig_afe_vs_noise.png")
    metric_vs_noise(
        plt, rows, "J_tau", 0.2, "Robot effort vs sensor noise (alpha = 0.2; C0 omitted, zero by definition)",
        "J_tau (N^2 m^2 s)", "fig_effort_vs_noise.png", log=False, skip=("C0",),
    )
    trace_figure(plt, traces, "fig_elbow_trace.png")
    print(f"wrote {len(rows)} trials to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
