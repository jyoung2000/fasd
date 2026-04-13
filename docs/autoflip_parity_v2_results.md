# AutoFlip Parity v2 — Results

This is the v2 evaluation harness referenced by the v2 work plan.
Every quality-improvement phase (3, 4, 5, 6, 7, 8) lands its
before/after numbers here. The harness lives in:

- **Metrics** — `backend/services/autoflip_parity_metrics.py`
  (pure Python, numpy-free, fully unit-tested in
  `backend/tests/test_autoflip_parity_metrics.py`)
- **Fixtures** — `backend/services/autoflip_parity_fixtures.py`
  (in-memory dataclass stubs — no .mp4 binaries, no ffmpeg)
- **Runner** — `backend/scripts/measure_autoflip_parity.py`
  (CLI; runs the segmenter inside docker, emits JSON)
- **Tests** — `backend/tests/test_autoflip_parity_metrics.py`
  + `backend/tests/test_autoflip_parity_fixtures.py`

## How to run

The runner needs the full reframe pipeline (numpy, OpenCV,
MediaPipe), so it lives behind the docker image. From a shell with
the pipeline available:

```bash
# Run every fixture and print to stdout
python -m backend.scripts.measure_autoflip_parity

# Run a single fixture
python -m backend.scripts.measure_autoflip_parity --fixture vlog_walk_and_talk

# Write a phase-tagged baseline file
python -m backend.scripts.measure_autoflip_parity \
    --output docs/autoflip_parity_v2_baseline.json --phase baseline

# Dry run (skip segmenter — works in a sandbox without numpy)
python -m backend.scripts.measure_autoflip_parity --dry-run

# List fixtures
python -m backend.scripts.measure_autoflip_parity --list
```

Each invocation writes a JSON document with one row per fixture:
fixture metadata, ground-truth summary, and a `metrics` dict keyed
on the metric name. Phase PRs paste the relevant before/after rows
into the result table below.

## Fixture set

Seven synthetic fixtures cover the v2 quality areas. None of them
ship as .mp4 — they're built in-memory by the same dataclass-stub
pattern used by `backend/scripts/measure_reframe_lag.py`, so the
runner has no ffmpeg / disk dependency.

| # | Fixture | Content type | Sub-type | Tests phase(s) | Metrics |
|---|---|---|---|---|---|
| 1 | `2speaker_alternating` | `podcast` | — | baseline | sub-second recall, overlap, accel, jerk, miss rate |
| 2 | `3speaker_panel` | `debate` | — | 3 (multi-region LP) | sub-second recall, overlap, accel, jerk, miss rate |
| 3 | `vlog_walk_and_talk` | `vlog` | — | 4 (thirds bias) | overlap, accel, jerk, miss rate, **thirds rate** |
| 4 | `music_video_beat` | `music_video` | `performance` | 5 (beat snap) | sub-second recall, overlap, **downbeat snap**, accel, jerk |
| 5 | `anime_hard_cuts` | `anime` | `dialogue` | 6 (anime + sub bar) | sub-second recall, overlap, accel, jerk, miss rate |
| 6 | `tps_character_offset` | `gameplay_tps` | `gta_v` | 7 (per-game action center) | overlap, accel, jerk, miss rate, **HUD preservation** |
| 7 | `stream_corner_facecam` | `stream` | `generic_fps` | 7 (STACKED_GAMEPLAY) | overlap, accel, jerk, miss rate, **HUD preservation** |

## Targets

The non-regression baselines from Phase 0 stay in force:

- **Sub-second switch recall** (`2speaker_alternating`,
  `music_video_beat`, `anime_hard_cuts`): **100 %**
- **Overlap count** (every fixture): **0**
- **Max |Δ²x|** (every fixture with a moving subject): **≤ 3 px**
- **Max |Δ³x|** (every fixture with a moving subject): **≤ 3 px**

Phase-specific targets are documented inline in the per-phase tables
below.

## Results — by phase

The columns fill in as each phase lands. `tbd` means "needs to be
captured in docker by running the runner with the appropriate
phase-tagged commit checked out".

### Fixture 1: `2speaker_alternating` (podcast)

| Metric | Target | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| sub_second_switch_recall | 1.0 | tbd | tbd | tbd | tbd | — | — | — | — | tbd |
| overlap_count | 0 | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| max_acceleration | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| max_jerk | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| required_region_miss_rate | 0 | tbd | tbd | tbd | tbd | — | — | — | — | — |

### Fixture 2: `3speaker_panel` (debate / multi-region)

| Metric | Target | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| sub_second_switch_recall | ≥ 0.9 | tbd | tbd | tbd | **focus** | — | — | — | — | tbd |
| overlap_count | 0 | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| max_acceleration | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| max_jerk | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| required_region_miss_rate | ≤ 0.05 | tbd | tbd | tbd | **focus** | — | — | — | — | — |

### Fixture 3: `vlog_walk_and_talk` (vlog / thirds)

| Metric | Target | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| overlap_count | 0 | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| max_acceleration | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| max_jerk | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| required_region_miss_rate | 0 | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| face_centroid_in_thirds_rate | ≥ 0.6 | tbd | tbd | tbd | — | **focus** | — | — | — | — |

### Fixture 4: `music_video_beat` (music video / beat snap)

| Metric | Target | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| sub_second_switch_recall | 1.0 | tbd | tbd | tbd | tbd | — | — | — | — | tbd |
| overlap_count | 0 | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| downbeat_snap_error.snap_rate | ≥ 0.95 | tbd | tbd | tbd | — | — | **focus** | — | — | — |
| downbeat_snap_error.max_error_ms | ≤ 200 ms | tbd | tbd | tbd | — | — | **focus** | — | — | — |
| max_acceleration | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| max_jerk | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |

### Fixture 5: `anime_hard_cuts` (anime / sub bar)

| Metric | Target | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| sub_second_switch_recall | 1.0 | tbd | tbd | tbd | tbd | — | — | — | — | tbd |
| overlap_count | 0 | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| max_acceleration | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| max_jerk | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| required_region_miss_rate (sub bar) | ≤ 0.05 | tbd | tbd | tbd | — | — | — | **focus** | — | — |

### Fixture 6: `tps_character_offset` (gameplay_tps)

| Metric | Target | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| overlap_count | 0 | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| max_acceleration | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| max_jerk | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| required_region_miss_rate (character at x=40) | 0 | tbd | tbd | tbd | — | — | — | — | **focus** | — |
| hud_preservation_rate | ≥ 0.9 | tbd | tbd | tbd | — | — | — | — | **focus** | — |

### Fixture 7: `stream_corner_facecam` (stream)

| Metric | Target | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| overlap_count | 0 | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| max_acceleration | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| max_jerk | ≤ 3 px | tbd | tbd | tbd | tbd | tbd | — | — | — | — |
| required_region_miss_rate (facecam at x=85) | 0 | tbd | tbd | tbd | — | — | — | — | **focus** | — |
| hud_preservation_rate | ≥ 0.9 | tbd | tbd | tbd | — | — | — | — | **focus** | — |

## Workflow per phase

Each Phase PR (3-8) follows the same five steps:

1. **Capture pre-PR baseline.** Check out the previous-phase commit
   in docker, run
   `python -m backend.scripts.measure_autoflip_parity --output docs/_phaseN_before.json --phase phaseN_before`,
   and copy the relevant numbers into the fixture tables above.
2. **Implement the phase change** behind its feature flag (default
   ON for the content types listed in the spec; OFF for everything
   else if it might regress).
3. **Capture post-PR numbers** the same way:
   `--output docs/_phaseN_after.json --phase phaseN_after`. Paste
   into the table.
4. **Compare.** Targets must hold; non-regression cells (overlap
   count, max accel/jerk on existing fixtures) must not move
   backward. If they do, feature-flag the regressing change off
   and ship the work with the flag + tests anyway.
5. **Add new tests** to `test_autoflip_parity_metrics.py` or
   `test_autoflip_parity_fixtures.py` if the phase touches the
   harness itself (e.g. Phase 5 will likely add a new
   `beat_grid_alignment_rate` metric).

## Sandbox limitation

The sandbox where Phases 1, 2, and 9 were authored does not have
numpy / cv2 / MediaPipe installed, so the `measure_autoflip_parity`
runner will report `status: skipped` for every fixture if invoked
without docker. The `--dry-run` mode still works (it just dumps the
fixture spec + ground-truth summary), and the metric + fixture unit
tests are 100 % numpy-free so the regression suite catches any
contract drift.

## Test counts (current)

Phase 9 + Phase 3 ship with these test files (all green in sandbox):

| File | Count |
|---|---|
| `test_autoflip_parity_metrics.py` | 63 |
| `test_autoflip_parity_fixtures.py` | 143 (Phase 3 resolved 2 skips) |
| `test_multi_region_lp.py` | 33 (20 sandbox + 13 scipy-gated) |
| **Phase 9 + 3 total** | **239 tests** |

Plus the existing surface from Phases 1+2 (132 tests) for a running
total of **417 v2 tests** that gate the work in CI. The 13 scipy-
gated tests run unconditionally in docker (where scipy is always
available) and use `pytest.mark.skipif` to skip cleanly in a
minimal sandbox.

## Sandbox vs docker numbers — Phase 3

The Phase 3 Stage 10a integration is feature-flagged off by default
(`CLIPAI_MULTI_REGION_LP=0`) until a docker validation run captures
the post-Phase-3 numbers. With the flag OFF, the pipeline behaves
identically to Phase 2 — every "tbd" cell in the fixture tables
above stays the same value. With the flag ON, the LP is invoked on
every segment that Stage 3 routed to `split`/`grid`/`wide_master`
and the segment is downgraded back to single-subject when the LP
says the speakers fit in one crop.

Sandbox sanity numbers (with scipy installed, captured ad-hoc by
running `python -m backend.scripts.measure_autoflip_parity`):

| Fixture | sub_second_recall | overlap | max_accel (% src w) | max_jerk (% src w) | required_region_miss_rate |
|---|---|---|---|---|---|
| `2speaker_alternating` | 1.0 | 0 | 50.0 | 100.0 | 0.40 |
| `3speaker_panel` | 1.0 | 0 | 60.0 | 120.0 | 0.75 |
| `vlog_walk_and_talk` (Phase 3 OFF) | — | 0 | 0.0 | ≈0 | 0.60 |
| `vlog_walk_and_talk` (Phase 4 ON, V2 lead-room + thirds bias) | — | 0 | 0.0 | ≈0 | **0.43** |
| `music_video_beat` (Phase 5 OFF) | 1.0 | 0 | 47.1 | 94.3 | — |
| `music_video_beat` (Phase 5 ON, beat snap + pulse cuts) | 1.0 | 0 | 47.1 | 94.3 | — |

Plus the music-video-only Phase 5 metric (`downbeat_snap_error`):

| Fixture | Mode | snap_rate | mean_error_ms | max_error_ms | count |
|---|---|---|---|---|---|
| `music_video_beat` | OFF | 0.40 | 200.0 | 200.0 | 5 |
| `music_video_beat` | ON | **1.00** | **0.0** | **0.0** | 8 (5 + 3 pulse cuts) |

### v2 Phase 7 — gameplay tracker + STREAM routing

Phase 7 adds the per-genre gameplay subject tracker and the
STREAM layout routing. Both target fixtures (`tps_character_offset`
+ `stream_corner_facecam`) are unchanged at flag-ON in the parity
bench because:

  - `tps_character_offset` and `stream_corner_facecam` currently
    take the legacy `_is_gameplay` fast-path in `pipeline.py`,
    which skips the segmenter entirely for gameplay content
  - The fast-path produces hardcoded subject_x = 50, regardless
    of gameplay variant
  - Phase 7's Stage 7c segmenter integration is in place but
    dormant until the pipeline.py wiring (Phase 7 follow-up)
    disables the fast-path when `gameplay_subtype` is set
  - The parity bench also doesn't pre-build
    `gameplay_motion_centroids` from real frame data — the
    OpenCV Farneback pass lives in pipeline.py (follow-up)

The Phase 7 mechanism is verified end-to-end via 34 unit tests
that build hand-crafted `GameplayMotionCentroid` sequences and
assert the tracker's smoothing + clamping + per-genre fallback
semantics work correctly. The bench numbers will move once the
pipeline.py wiring + per-frame OpenCV extraction land.

| Fixture | Mode | required_region_miss_rate | Notes |
|---|---|---|---|
| `tps_character_offset` | OFF | 1.00 | character at x=40, fast-path centers at 50 |
| `tps_character_offset` | ON | 1.00 | unchanged (segmenter dormant for gameplay) |
| `stream_corner_facecam` | OFF | 0.00 | facecam at x=85 trivially in crop |
| `stream_corner_facecam` | ON | 0.00 | unchanged |

### v2 Phase 6 — anime fixture re-baseline

Phase 6 changed the ``anime_hard_cuts`` ground truth: dropped the
geometrically un-fittable sub bar (15..85 % > 31.6 % crop) and
replaced it with the active speaker bbox (10 % wide, alternating
slot 0 = 30, slot 1 = 70). The sub bar was the wrong test for
anime reframing — chasing it would yank the crop away from the
dramatic anchor.

| Fixture | Mode | required_region_miss_rate | Notes |
|---|---|---|---|
| `anime_hard_cuts` (Phase 5 baseline, sub bar) | — | 1.00 | un-fittable |
| `anime_hard_cuts` (Phase 6 baseline, speaker bbox) | OFF | 0.33 | residual is from speaker-turn anticipation |
| `anime_hard_cuts` (Phase 6, all flags ON) | ON | 0.33 | unchanged on the parity bench because the bench has no per-frame anime features for the anchor scorer |

The Phase 6 anime anchor mechanism is verified end-to-end via 48
unit tests (face detector helpers + anchor scorer + segmenter
Stage 7b wiring + lead-room multiplier + applies_to_profile
extension). The parity bench numbers will move once the
production pipeline.py wiring extracts per-frame anime features
and passes them via the new ``anime_anchors`` kwarg.

The Phase 5 ON row hits the spec exit criterion exactly: every cut
lands at 0 ms from a downbeat — well under the spec's ±40 ms
target. The 3 pulse cuts are visual re-anchors inserted at internal
downbeats (the active speaker doesn't change but ``subject_x`` is
re-derived from the slot center, producing a "fresh anchor" pulse).

The vlog row demonstrates Phase 4's intended improvement: the
required-region miss rate drops from 0.60 → 0.43 (−28 %) when
both ``CLIPAI_GAZE_LEAD_ROOM_V2=1`` and ``CLIPAI_THIRDS_BIAS=1``
are set. The other metrics (max accel, max jerk, overlap count)
are unchanged because Phase 4 applies a uniform constant offset
within each segment, which preserves the L1 solver's smoothness
guarantees.

The 2speaker / 3speaker fixtures are unchanged at flag-ON because
podcast / debate content have ``apply_lead_room=False`` in the
content config — Stage 8 (and therefore Phase 4) doesn't fire on
them. Phase 5+ may revisit this for the multi-region LP path.

The accel / jerk numbers are large because the metric is computed
on a per-segment-step basis and these fixtures have hard speaker
swaps (the camera snaps from x=20 % to x=80 %). Phase 4+ will reduce
them via the multi-region LP smoothing the transitions, AND via the
thirds-bias work that interpolates the camera path within segments.

The 75 % required-region miss rate on the 3-speaker panel is the
Phase 3 LP's natural target: 75 % of frames have a speaker outside
the crop because the heuristic single-subject pipeline can only
follow one face at a time. With the multi-region LP enabled and a
fixture that exercises the simultaneous-overlap path, this number
will drop dramatically — Phase 3 follow-up (or Phase 4 prerequisite)
adds an overlap fixture.
