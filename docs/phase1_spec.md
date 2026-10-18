# Phase 1 Specification (DRAFT)

Status: draft for review. Companion to Project Charter v3.0 (`Synergy_Aware_Assist_Controller_Project_Charter_v3.pdf`).
Purpose: freeze coordinates, units, equations and definitions before any simulator code is written.

Items marked **[TBD-CAL]** are numeric values fixed during the calibration phase, then frozen (charter section 19).

## 1. Conventions

- Units: SI throughout (m, kg, s, rad, N·m). Angles in radians internally.
- Horizontal-plane arm, gravity neglected in the plane (charter section 8).
- Shoulder at the origin. `q_s` = shoulder angle from the +x axis, counter-clockwise positive.
- `q_e` = elbow angle relative to the upper arm. **Flexion is positive**, `q_e = 0` is full extension. Elbow range `[0, q_e,max]`.
- State: `q = [q_s, q_e]`, `qdot`. Hand position `x = fk(q)`.
- `u_abd` in `[0, 1]`: normalised shoulder-abduction demand. A virtual effort/load input, not a joint angle. Constant within a trial in the primary factorial experiment (static-loading interpretation). A separate dynamic-`u_abd` experiment (step and ramp profiles) tests estimator tracking and latency only; it is not part of the primary comparison (section 8). *Decision 1: resolved.*

## 2. Arm dynamics

`M(q) qddot + C(q, qdot) qdot + B qdot = tau_human + tau_robot`

- Two-link planar manipulator, standard closed-form `M` and `C` (Siciliano et al., ch. 8-9). Several Coriolis matrices give the same vector `C(q, qdot) qdot`; the implementation fixes one convention (document it in the code), and tests validate the resulting Coriolis/centrifugal **vector** and the dynamics, not a particular matrix form.
- Segment masses, lengths, centre-of-mass offsets, inertias: **[TBD-CAL]**, to be taken from published anthropometric tables and cited. Do not tune for appearance.
- `B`: small viscous damping, justified and documented.
- Integration: fixed step, RK4. Timestep and controller rate **[TBD-CAL]** (proposal: dt = 1 ms, controller at 1 kHz). Convergence test required (section 9).

## 3. Reaching task

- Fixed start posture, targets at a fixed distance in at least three directions.
- Reference: minimum-jerk hand trajectory of fixed duration `T`, mapped to joints by inverse kinematics, giving `q_d(t)`, `qdot_d(t)`. Distance and `T` **[TBD-CAL]**.
- Reach counts as complete if final endpoint error is below a tolerance at time `T` **[TBD-CAL]**.

## 4. Simulated human

`tau_human = tau_vol + tau_syn`, kept separate at all times.

### 4.1 Voluntary motor model (Decision 2: resolved)

`tau_vol = sat( tau_ff + K_h (q_d - q) + D_h (qdot_d - qdot) )`

- Saturation is applied to the **total** voluntary command, per joint and per direction: `tau_flex,max,j` and `tau_ext,max,j`. Limits are direction-specific, so the extension limit at the elbow is named `tau_ext,max,e` and is distinct from the flexion limit.
- `tau_ff`: inverse-dynamics feedforward for the *reference* using the nominal arm model. It does **not** include `tau_syn`, so the simulated person is unaware of the impairment and does not compensate by construction.
- `K_h, D_h`: human feedback gains, deliberately lower than the robot's, so the robot has a real job.
- The finite limits are what make the reachable workspace shrink under load (section 9, Level 3).
- Not modelled in v1: active reduction of effort when assisted, fatigue, adaptation. State this as a limitation in the report.

### 4.2 Impairment torque

`tau_syn = [0, alpha * u_abd * tau_scale]` (elbow channel only, flexion direction), initial linear form `f(u) = u` as in charter section 10.

- Non-dimensionalisation (Decision 5: resolved): **`tau_scale = tau_ext,max,e`**, the elbow's maximum voluntary *extension* torque, which is the capacity that opposes the flexion-directed `tau_syn`. `alpha` is a normalised severity: the fraction of available elbow extension torque the synergy demands at full abduction demand. Do not define it through a "can just hold position" claim; whether the arm holds depends on posture-dependent terms and the other torques in play.
- Severity set: `alpha` in `{0, mild, moderate, severe}`; values **[TBD-CAL]**, to be chosen so the three levels produce clearly distinct workspace loss without total failure at "severe".
- Note: `tau_syn` is non-zero at rest when `u_abd > 0`. This is intended (static abduction loading) but means the baseline impedance controller will partly cancel it. Report the C0 vs C1 gap explicitly so the controller comparison is not trivially small.

## 5. Robot controllers

All controllers are subject to torque and torque-rate limits (charter section 14).

| ID | Name | Definition |
|----|------|-----------|
| C0 | Unassisted | `tau_robot = 0` |
| C1 | Conventional impedance | `tau_base = K_p (q_d - q) + K_d (qdot_d - qdot)`, tuned once under `alpha = 0` |
| C2 | Generic elbow assistance | C1 plus extra elbow-only stiffness/damping, always on, no detection and no use of `u_abd`. Matched budget: the gain rule (a mapping from condition to extra elbow gain that equalises mean `J_tau` with the headline C3) is fitted on the **calibration set only**, frozen, then applied unchanged to the final set. It is never re-fitted using final-set C3 results |
| C3 | Synergy-aware (headline) | C1 plus estimated-synergy cancellation `tau_assist,e = -alpha_hat * u_abd * tau_scale` (extension direction), with `alpha_hat` from the estimator (section 6), active only when `u_abd > u_min` and the estimate is valid |
| C3-fb | Ablation | C1 plus residual feedback `tau_assist,e = -K_c e_hat - D_c d(e_hat)/dt`, active only above a detection threshold and when `u_abd > u_min` (the form in charter section 13) |
| C-oracle | Upper bound | C1 plus `-alpha * u_abd * tau_scale` using the **true** `alpha`. Uses ground truth, so it is not a realistic controller; label it as such in every figure |

Decision 3: resolved. The headline C3 is estimated-synergy cancellation. The comparison ladder is: no help (C0), conventional help (C1), generic extra elbow help (C2), residual feedback on the estimate (C3-fb), estimated cancellation (C3), true cancellation (C-oracle). C3-fb is structurally close to C2, so C3 vs C3-fb tests whether using the load relationship (`u_abd`) is what adds value. The charter's section 13 formulation is retained as C3-fb; the charter should be amended to match when it is next revised.

## 6. Synergy estimator

Inputs allowed: measured `q`, `qdot` (with simulated noise), `q_d`, commanded `u_abd`, past robot torques. Not allowed: true `tau_syn`, true `alpha`, anything from the ground-truth channel. Enforce by interface (the estimator function receives a restricted observation object) and by a test.

- Online expected elbow: `q_e,expected,online(t) = q_d,e(t)`, the planned reference elbow angle. Residual `e_syn = q_e - q_d,e`.
- Known confound: under C1 the residual also contains ordinary tracking error. Handle by (a) causal windowed averaging as in charter 12.2, (b) gating on `u_abd`, and (c) measuring the residual's false-positive rate on `alpha = 0` trials across all directions (RQ2).
- Severity estimate `alpha_hat` (drives C3). Model-based, from the windowed residual `e_hat` (charter 12.2):
  `d_hat_e = K_eff,e * e_hat`, with `K_eff,e = K_p,e + K_h,e` (nominal elbow stiffnesses of robot and simulated human), then
  `alpha_hat = clip( d_hat_e / (u_abd * tau_scale), 0, alpha_max )`, defined only when `u_abd > u_min`.
  This is a quasi-static relation: the elbow settles where the restoring stiffness balances the synergy torque. It is least accurate during fast motion (inertial terms) and when the human command saturates (the human stiffness is then not delivered). Both are known limitations; a dynamic disturbance-observer form is a candidate refinement **[TBD-CAL]**.
- The estimator uses nominal `K_h`. Mismatch between the estimator's `K_h` and the simulated human's true `K_h` is a named robustness test (Level 4), since `alpha_hat` accuracy depends on it directly.
- Keep `u_abd` observability explicit: the estimator receives the commanded `u_abd` as an input (charter 12.2); it does not need to infer it.
- Outputs: `e_hat`, `alpha_hat`, detection state, confidence, latency. Fallback to C1 if the estimate is invalid (charter section 14).
- Dynamic-`u_abd` test (section 8) measures how quickly `alpha_hat` converges after a step, since the window length trades noise against latency.

## 7. Metrics

Defined as in charter section 16, with these precisions.

- **`q_e,expected` for the metric (offline)**: the elbow trajectory from the same trial (same target, same `u_abd`, same human) with `alpha = 0` under **C1**. One reference per `(target, u_abd)` condition, shared by all controllers, so no controller is compared against its own counterfactual. These reference trajectories are generated and frozen before C2/C3 are evaluated, so the benchmark cannot depend on their results. *Decision 4: resolved.* This differs deliberately from the online definition in section 6, to avoid the detector and the metric sharing a definition.
- `E_AFE = (1/T) * integral of max(0, q_e - q_e,expected)^2 dt` (flexion excess only).
- `J_tau = integral of ||tau_robot||^2 dt`.
- Smoothness: dimensionless squared jerk of the hand path (Hogan & Sternad, 2009 - verify citation details before use). Report the filtering and differentiation method.
- `G_syn` (charter 16.3) is `alpha * tau_scale` by construction in this model; record it as a ground-truth label, not as a result.
- Detector: estimation error against ground truth, latency, false positives (alpha = 0), false negatives.
- Failed trials are kept and reported.

## 8. Experimental design

- Factors: severity {0, mild, moderate, severe} x `u_abd` {low, medium, high} x direction (>= 3) x controller {C0, C1, C2, C3-fb, C3, C-oracle}.
- Dynamic-`u_abd` experiment (separate from the above): step and ramp profiles at one or two severities, evaluating estimator tracking error and latency only. Not used for the controller comparison.
- Noise: Gaussian on `q`, `qdot`; levels **[TBD-CAL]**.
- Calibration set vs final set are separate. Thresholds and gains are chosen on the calibration set and frozen before the final set is run.

## 9. Verification and validation plan

Mapped to charter section 18 and to files under `tests/`.

| Level | Tests |
|-------|-------|
| 1 Numerical | zero-input stays at rest; open-loop torque gives expected acceleration (`M^-1 tau` at rest); energy non-increasing with damping and no input; dt convergence; joint limits; unit tests for `fk`, `ik`, `M`, `C` |
| 2 Internal | `alpha = 0` gives zero `tau_syn`; `u_abd = 0` gives zero `tau_syn`; `tau_syn` rises monotonically in `alpha` and `u_abd`; `tau_syn` acts only on the elbow in the flexion direction |
| 3 Phenomenological (run under C0, and separately under C1; never under C3, whose purpose is to counteract the effect) | reachable-workspace fraction decreases monotonically with `u_abd` for fixed `alpha > 0` (workspace = set of grid targets reached within tolerance under `tau_vol,max`); elbow flexion excess rises with `u_abd` and with `alpha`; report as direction-of-effect only |
| 4 Robustness | altered arm parameters, unseen `alpha` and directions, sensor noise, altered `u_abd` |
| Interface | estimator cannot access ground truth (test passes only the restricted observation object) |

## 10. Decisions

All five Phase 1 decisions are resolved:

1. Constant `u_abd` in the primary experiment; step/ramp as a separate estimator test.
2. Voluntary model: nominal feedforward unaware of the synergy, weak feedback, saturation on the total command.
3. Headline C3 = estimated-synergy cancellation; residual feedback (C3-fb) as ablation; true-`alpha` oracle as upper bound.
4. Metric reference = `alpha = 0` under C1, shared across controllers and frozen before C2/C3 evaluation.
5. `tau_scale = tau_ext,max,e`.

Remaining before code: the **[TBD-CAL]** numeric values (anthropometrics, gains, limits, severity levels, noise), set in the calibration phase and then frozen.
