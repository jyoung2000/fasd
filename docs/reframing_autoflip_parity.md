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

## Saliency Parity Fixes (Fixes 1-6)

Six fixes to bring saliency signal processing to AutoFlip parity:

1. **Wire full bboxes** — `scene_focus` now receives real `(x, y, w, h)` from SaliencyRegion instead of hardcoded `y=50, w=10, h=10`
2. **Unconditional saliency** — removed face-gated skip; saliency runs on every frame like AutoFlip
3. **Adaptive threshold** — percentile-based (top 15%) instead of fixed 0.5; low-contrast shots now produce regions
4. **Center bias + HUD masking** — 2D Gaussian center prior (σ=0.35) suppresses background motion; HUD mask zeros out known HUD pixels
5. **Feature merging** — overlapping saliency+face bboxes (IoU>0.3) are absorbed to prevent over-constraining the crop solver
6. **Temporal smoothing** — EMA (α=0.3) on promoted saliency cluster trajectories reduces jitter

Tests: 27 → 37 passing (+10 new tests across 3 test files)

---

## Out of scope (follow-ups)

- Multi-region tracking (AutoFlip handles multiple required regions with a unified LP) — **queued for Phase 3 of v2**
- Cinematography "rule of thirds" bias — **queued for Phase 4 of v2**
- Gameplay HUD-aware padding (orthogonal to solver quality, lives in layout_engine)

---

## v2 — Content-aware parity work (Phases 1–10)

The "Phases 0–4" table above covers the solver-quality work. A second
wave of phases starts from the opposite end: routing per-content-type
behaviours so anime, debates, music videos, vlogs, and multi-game
streams all get editorially-correct framing instead of falling into
the "generic talking-head" bucket. This section will grow one sub-
heading per v2 phase as they land.

### v2 Phase 1 — Content-type override plumbing

**Before:**
The upload UI dropdown sent strings (`"gameplay"`, `"movie"`,
`"podcast"`) that didn't match the `ContentType` enum values
(`"gaming"`, `"narrative"`, `"podcast"`). The classifier's
user-override branch at `content_classifier.py:122-130` checked
`user_type in [ct.value for ct in ContentType]`, so:

| UI token | Enum value | Override fired? |
|---|---|---|
| `gameplay` | `gaming` | ❌ |
| `movie` | `narrative` | ❌ |
| `podcast` | `podcast` | ✅ (by coincidence) |

Worse, `metadata['content_type_override']` was never injected into
the dict `classify_content` received — the ffprobe metadata dict
historically did NOT carry that key, so even the `"podcast"` case
only worked if a downstream caller happened to plumb the override
through as `metadata['content_type']` or `metadata['reframe_style']`,
which the pipeline did not.

Gameplay worked only via a separate hard-coded
`_content_override == "gameplay"` comparison at `pipeline.py:1226`,
which bypassed the classifier entirely — so downstream tuning still
thought content type was `UNKNOWN` even for user-declared gameplay.

**After (v2 Phase 1):**

- **New module** `backend/services/content_type_strings.py` with
  `normalize_ui_content_type(token)`, `is_gameplay_override(token)`,
  and `is_user_override(token)`. Maps every UI token (both legacy and
  Phase 2 forward-compat) to a
  `NormalizedContentType(content_type, is_multi_speaker_panel,
  is_animated, is_gameplay_fastpath, raw)`.
- **Classifier override branch** now normalizes through the helper,
  honours `is_multi_speaker_panel` / `is_animated` on the profile,
  and logs the normalized form. Accepts `content_type_override`,
  `content_type`, and `reframe_style` metadata keys so older callers
  still work.
- **Pipeline** injects `_content_override` into a `_classifier_metadata`
  dict before calling `classify_content`, so the UI override actually
  reaches the classifier's user-override branch. Gameplay fast-path
  now uses `is_gameplay_override()` instead of a bare string literal,
  so the Phase 2 gameplay variants (`gameplay_fps` / `gameplay_moba`
  / `gameplay_tps` / `gameplay_racing`) are covered without further
  edits here. Non-gameplay user overrides now correctly skip the
  gameplay auto-detect step via `is_user_override()`.
- **`stream`** normalizes to `ContentType.GAMING` but is deliberately
  NOT a gameplay fast-path candidate — it has a facecam and still
  needs the face pipeline. Phase 7 will handle its `STACKED_GAMEPLAY`
  layout routing downstream.

### Normalization matrix

| UI token | `ContentType` | Flags | fastpath |
|---|---|---|---|
| `gameplay` *(legacy)* | `gaming` | — | yes |
| `movie` *(legacy)* | `narrative` | — | no |
| `podcast` *(legacy)* | `podcast` | — | no |
| `debate` / `panel` | `podcast` | panel | no |
| `interview` | `podcast` | — | no |
| `vlog` | `vlog` | — | no |
| `narrative` / `cinematic` | `narrative` | — | no |
| `anime` / `cartoon` | `anime` | animated | no |
| `music_video` | `music_video` | — | no |
| `gameplay_fps` | `gaming` | — | yes |
| `gameplay_moba` | `gaming` | — | yes |
| `gameplay_tps` | `gaming` | — | yes |
| `gameplay_racing` | `gaming` | — | yes |
| `stream` | `gaming` | — | **no** (facecam) |
| `sports` | `sports` | — | no |
| `auto` / `""` / invalid | — | — | — (heuristic path) |

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_content_type_override_plumbing.py` | 78 | normalizer / `classify_content` override branch / `pipeline.py` AST plumbing |
| Pre-existing classifier/plumbing suites | 54 | zero regressions |
| **Total (v2 Phase 1 scope)** | **132** | all green |

### Exit criteria met

- ✅ Every UI token round-trips through `normalize_ui_content_type` to
  the correct `ContentType` enum value.
- ✅ `classify_content` with `metadata['content_type_override']`
  short-circuits to `confidence=1.0` on the normalized enum value and
  sets `is_multi_speaker_panel` / `is_animated` from the normalized
  bundle.
- ✅ Pipeline AST asserts the metadata injection is present and that
  the gameplay fast-path uses `is_gameplay_override()` instead of a
  bare `== "gameplay"` literal.
- ✅ Invalid / unknown tokens fall through to the heuristic path
  without crashing; `confidence != 1.0` so downstream consumers can
  still distinguish a user override from a heuristic guess.
- ✅ Legacy `metadata['content_type']` / `metadata['reframe_style']`
  keys still work for older callers.
- ✅ No regression on the 54 pre-existing classifier / plumbing
  tests; no regression on `test_pipeline_source_width_hoisted.py`
  (which guards that `source_width` is still bound at function
  scope despite the nearby edits).
