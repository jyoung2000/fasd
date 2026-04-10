# AutoFlip-Parity Reframing Changelog

Tracking progress toward full parity with Google AutoFlip
(Grundmann et al., "Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths").

## Baseline (pre-Phase 0)

| Metric | Value | Notes |
|---|---|---|
| Detected speaker changes | 9/9 (100%) | Synthetic 20s fixture, alternating every 2s |
| Missed changes | 0 | |
| Avg lag (frames @ 30fps) | 6.0 | Negative = anticipation (-6 frames = -200ms) |
| Max lag (frames) | 6 | |
| Min lag (frames) | 6 | |
| Sub-second switch recall | 100% (4/4) | 5 alternating 400ms segments |
| Overlap count | 0 | |
| Smoke test | PASSED | 1 segment, strategy=panning, 10s |
| L1 solver tests | 12/12 passed | test_l1_solver_constraints.py |

### Known issues (mapped to phases)

1. **Jitter during holds** — solver only penalizes velocity (no accel/jerk terms)
2. **2 Hz stair-stepping** — solver receives sparse anchors, not dense propagated trajectory
3. **~1s lag on transitions** — per-segment solving with no lookahead
4. **Pops at segment seams** — segments solved independently
5. **Chases keypoint noise** — TV_LAMBDA=10.0 too low, no dead-zone

---

## Phase 0 — Baseline telemetry

- Added CSV dump behind `CLIPAI_DUMP_L1=1` env flag → `/tmp/clipai_l1/<job_id>_<seg_idx>.csv`
- Added `backend/scripts/plot_l1_solve.py` for target-vs-solved visualization
- Baseline numbers recorded above

---

## Phase 1 — Dense propagated trajectory

**Fix:** Feed the solver the dense propagated trajectory at uniform fps instead of sparse ~2Hz anchors.

- Added `get_propagated_positions_for_segment` with linear interpolation + last-known hold
- Resamples onto uniform grid at `target_fps=30.0` — critical for TV solver's finite-difference operator
- Updated reframe_segmenter Stage 10 and pipeline to thread `interpolated_timeline`
- 5 unit tests added (sine wave uniform, gap handling, empty, single, solver integration)

| Metric | Before | After | Change |
|---|---|---|---|
| All lag metrics | unchanged | unchanged | No change expected (lag fix is Phase 3) |
| Smoke test | PASSED | PASSED | subject_x shifted slightly due to resampling |
| Tests | 12/12 | 17/17 | +5 new tests |

---

## Phase 2 — Exact TV-L2 solver with dead-zone

**Fix:** Replace 100-iteration proximal-gradient TV with exact dual proximal gradient solver.

- New `_condat_tv.py`: exact L2-TV proximity operator (Chambolle 2004 / Condat 2013 dual formulation)
- Converges to machine precision — zero residual tilt on constant signals
- Resolution-independent lambda: `TV_LAMBDA_FRAC=0.015 * source_width` (28.8 at 1920)
- Pre-solve dead-zone: suppress keypoint jitter < 25px (DEADZONE_FRAC=0.013)
- 4 unit tests added (constant, step, bounds, smoother)

| Metric | Before | After | Change |
|---|---|---|---|
| Residual tilt on holds | >0 (approximate solver) | 0 (exact) | Eliminated |
| Smoke test | PASSED | PASSED | |
| Tests | 17/17 | 21/21 | +4 new tests |

---

## Phase 3 — Shot-level solving with lookahead

**Fix:** Solve L1 camera path once per shot (not per segment), with mirror-reflection lookahead padding.

- New `solve_camera_path_for_shot`: concatenates segment targets, solves once, slices back
- Mirror-reflects 30 frames at shot boundaries for symmetric non-causal smoothing
- Reframe_segmenter Stage 10 groups segments by `shot_cuts` and calls shot-level solver
- `solve_camera_path` kept as backward-compatible thin wrapper

| Metric | Before | After | Change |
|---|---|---|---|
| Lookahead proven | N/A | Solved path starts moving BEFORE subject transition at t=2.0s | New capability |
| Segment boundary continuity | Pops possible | <5px gap at boundaries | Eliminated |
| Smoke test | PASSED | PASSED | |
| Tests | 21/21 | 25/25 | +4 integration tests |

---

## Phase 4 — LP solver with acceleration and jerk penalties

**Fix:** Add LP-based AutoFlip solver with λ₃ (acceleration) and λ₄ (jerk) penalties.

- New `_autoflip_lp.py`: formulates as LP with HiGHS backend (scipy.optimize.linprog)
- Each |·| term linearized via auxiliary slack + two inequality constraints
- Feature flag `CLIPAI_L1_LP=1` (ships OFF by default for A/B testing)
- Performance guardrail: falls back to Condat for shots > 600 frames (20s)
- Comparison script `backend/scripts/compare_l1_solvers.py`
- Added scipy and numpy to requirements.txt

| Metric | Condat TV | AutoFlip LP | Improvement |
|---|---|---|---|
| Velocity (TV) | 1100.0 | 1100.0 | Same |
| Acceleration | 2195.9 | 53.7 | **41x less** |
| Jerk | 4391.6 | 3.7 | **1187x less** |
| Solve time (150 frames) | 24ms | 40ms | Acceptable |
| Solve time (300 frames) | ~50ms | ~100ms | Within 200ms budget |
| Tests | 25/25 | 29/29 | +4 LP tests |

---

## Final state

| Metric | Baseline | After all phases |
|---|---|---|
| Tests passing | 12 | 29 |
| Solver type | 100-iter prox-grad (approximate) | Exact dual proximal gradient + optional LP |
| Input to solver | Sparse ~2Hz anchors | Dense uniform 30fps trajectory |
| Solve scope | Per-segment | Per-shot with lookahead |
| Residual tilt on holds | Present | Zero |
| Acceleration penalties | None | λ₃ + λ₄ via LP (behind feature flag) |
| Dead-zone | None | 25px (resolution-scaled) |
| Lambda | Fixed 10.0px | Resolution-independent 0.015 * source_width |

---

## Out of scope (follow-ups)

- Multi-region tracking (AutoFlip handles multiple required regions with a unified LP)
- Cinematography "rule of thirds" bias
- Gameplay HUD-aware padding (orthogonal to solver quality, lives in layout_engine)
