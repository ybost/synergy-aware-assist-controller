"""CALIBRATION of C3's estimator window and cancellation ramp-in (placeholder parameters).

Selection rule, fixed BEFORE running (do not edit after seeing results):

  * Grid: window in WINDOWS samples x ramp in RAMPS control steps.
  * Calibration conditions use severities and noise seeds that the final comparison does
    not use (CAL_ALPHAS, CAL_SEEDS), so the choice cannot be fitted to evaluation data.
  * A grid point is FEASIBLE if, at every noise level, C3's mean dimensionless jerk over the
    reach is no worse than conventional C1's on the same trials (charter 19.3: no material
    deterioration in smoothness).
  * Among feasible points, pick the one with the smallest worst-case-over-noise-levels mean
    E_AFE. If none is feasible, report that and select nothing.

The parameters are placeholders, so the selected values are provisional until the real
calibration phase; the procedure is what carries over.

Usage:  python scripts/calibrate_c3.py [--quick]
Outputs: results/calibration/{c3_grid.csv, c3_selected.json, fig_c3_grid.png}
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from synergy_aware import placeholders as ph
from synergy_aware.controllers import ImpedanceController, SynergyAwareController
from synergy_aware.estimator import SynergyEstimator
from synergy_aware.metrics import (
    ReferenceSet,
    dimensionless_jerk,
    excess_elbow_flexion,
    hand_positions,
    robot_effort,
)
from synergy_aware.simulation import NoiseConfig, run_trial

OUT = Path(__file__).resolve().parents[1] / "results" / "calibration"
WINDOWS = (10, 20, 50, 100, 200)
RAMPS = (0, 50, 100, 200, 400)
CAL_ALPHAS = (0.15, 0.25)  # the final comparison uses 0.1, 0.2, 0.3
CAL_SEEDS = (200, 201)  # distinct from the C2 fit seed (100) and the evaluation seeds (0, 1, 2)
U_ABD = 1.0
U_MIN = 0.05


def noise_cfg(level, seed):
    spec = ph.NOISE_LEVELS[level]
    return None if spec is None else NoiseConfig(q_sd=spec[0], qd_sd=spec[1], seed=seed)


def trials(quick):
    alphas = CAL_ALPHAS[:1] if quick else CAL_ALPHAS
    levels = ("none", "medium") if quick else tuple(ph.NOISE_LEVELS)
    for a in alphas:
        for lv in levels:
            for s in CAL_SEEDS[:1] if (quick or lv == "none") else CAL_SEEDS:
                yield a, lv, s


def score_trial(res, cfg, ref_qe):
    win = int(round(cfg.T / cfg.dt)) + 1
    return {
        "E_AFE": excess_elbow_flexion(res, ref_qe),
        "DJ": dimensionless_jerk(hand_positions(res)[:win], cfg.dt),
        "J_tau": robot_effort(res, cfg.dt),
        "endpoint_mm": res.endpoint_error() * 1e3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    human = ph.human()
    ts = human.tau_max_neg[1]
    c1 = ImpedanceController(ph.c1_gains())
    refs = ReferenceSet(c1)
    conds = list(trials(args.quick))
    cfgs = {c: ph.base_cfg(alpha=c[0], u_abd=U_ABD, noise=noise_cfg(c[1], c[2])) for c in conds}
    refq = {c: refs.get(cfgs[c]) for c in conds}
    c1_scores = {c: score_trial(run_trial(cfgs[c], c1), cfgs[c], refq[c]) for c in conds}
    levels = sorted({c[1] for c in conds}, key=list(ph.NOISE_LEVELS).index)

    windows, ramps = (WINDOWS[:2], RAMPS[:2]) if args.quick else (WINDOWS, RAMPS)
    grid = []
    for w in windows:
        for r in ramps:
            per_level = {lv: {"E_AFE": [], "DJ": [], "J_tau": [], "DJ_c1": []} for lv in levels}
            for c in conds:
                ctl = SynergyAwareController(c1, SynergyEstimator(ts, human.K[1], w, U_MIN), ts, ramp_steps=r)
                s = score_trial(run_trial(cfgs[c], ctl), cfgs[c], refq[c])
                for k in ("E_AFE", "DJ", "J_tau"):
                    per_level[c[1]][k].append(s[k])
                per_level[c[1]]["DJ_c1"].append(c1_scores[c]["DJ"])
            mean = {lv: {k: float(np.mean(v)) for k, v in d.items()} for lv, d in per_level.items()}
            feasible = all(mean[lv]["DJ"] <= mean[lv]["DJ_c1"] for lv in levels)
            worst = max(mean[lv]["E_AFE"] for lv in levels)
            grid.append(dict(window=w, ramp=r, feasible=feasible, worst_E_AFE=worst,
                             worst_DJ_ratio=max(mean[lv]["DJ"] / mean[lv]["DJ_c1"] for lv in levels),
                             mean_J_tau=float(np.mean([mean[lv]["J_tau"] for lv in levels]))))
            print(f"window={w:3d} ramp={r:3d}  worst E_AFE={worst:.2e}  max DJ/DJ_C1={grid[-1]['worst_DJ_ratio']:.2f}  feasible={feasible}", flush=True)

    feas = [g for g in grid if g["feasible"]]
    selected = min(feas, key=lambda g: (g["worst_E_AFE"], g["mean_J_tau"])) if feas else None
    with open(OUT / "c3_grid.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(grid[0]))
        wr.writeheader()
        wr.writerows(grid)
    with open(OUT / "c3_selected.json", "w") as f:
        json.dump({"selected": selected, "n_feasible": len(feas), "n_grid": len(grid),
                   "note": "placeholder parameters; provisional", "cal_alphas": CAL_ALPHAS, "cal_seeds": CAL_SEEDS}, f, indent=2)
    print("selected:", selected)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])
    ws, rs = sorted({g["window"] for g in grid}), sorted({g["ramp"] for g in grid})
    vals = np.array([[next(g for g in grid if g["window"] == w and g["ramp"] == r)["worst_E_AFE"] for r in rs] for w in ws])
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(np.log10(vals), cmap=cmap, aspect="auto", origin="lower")
    for i, w in enumerate(ws):
        for j, r in enumerate(rs):
            g = next(g for g in grid if g["window"] == w and g["ramp"] == r)
            dark = np.log10(vals[i, j]) > np.log10(vals).mean()
            ax.text(j, i, f"{g['worst_E_AFE']:.1e}" + ("" if g["feasible"] else "\nnot smooth"), ha="center", va="center", fontsize=7,
                    color="#fcfcfb" if dark else "#0b0b0b")
            if selected and (w, r) == (selected["window"], selected["ramp"]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=3))
    ax.set_xticks(range(len(rs)), [str(r) for r in rs])
    ax.set_yticks(range(len(ws)), [str(w) for w in ws])
    ax.set_xlabel("cancellation ramp-in (control steps)", fontsize=10, color="#52514e")
    ax.set_ylabel("estimator window (samples)", fontsize=10, color="#52514e")
    ax.set_title("C3 calibration: worst-case E_AFE over noise levels", loc="left", fontsize=12, color="#0b0b0b")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log10 worst-case E_AFE (rad^2)", fontsize=9, color="#52514e")
    fig.text(0.01, 0.01, "CALIBRATION on placeholder parameters, provisional. Orange outline = selected; 'not smooth' = worse jerk than C1.", fontsize=7.5, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_c3_grid.png", facecolor="#fcfcfb")


if __name__ == "__main__":
    sys.exit(main())
