#!/usr/bin/env python3
"""Phase 9 — Measure AutoFlip parity across the v2 fixture set.

For each fixture in ``backend.services.autoflip_parity_fixtures`` this
script:

  1. Builds the in-memory inputs (no ffmpeg / no .mp4 binaries).
  2. Runs ``build_reframe_segments`` with the fixture's
     ``content_type_override`` injected via metadata so the Phase 1
     plumbing fires and the per-content-type tuning kicks in.
  3. Extracts the actual crop centers and switch times from the
     resulting segments.
  4. Scores them against the fixture's ground truth using the pure-
     Python metric library in ``backend.services.autoflip_parity_metrics``.
  5. Emits a JSON document with one row per fixture × metric.

The metric library is fully unit-testable in a numpy-less sandbox;
this runner needs the full reframe pipeline (which transitively needs
numpy / cv2 / MediaPipe) so it is intended to run inside the docker
image. A ``--dry-run`` mode skips the segmenter call and just reports
the fixture spec + ground truth, which is useful for verifying the
scaffolding in a sandbox.

Usage:
    # Run every fixture and print JSON to stdout
    python -m backend.scripts.measure_autoflip_parity

    # Run a single fixture
    python -m backend.scripts.measure_autoflip_parity --fixture vlog_walk_and_talk

    # Write to a phase-tagged baseline file
    python -m backend.scripts.measure_autoflip_parity \
        --output docs/autoflip_parity_v2_phase1.json --phase phase1

    # Dry run (no segmenter — just dump fixture specs)
    python -m backend.scripts.measure_autoflip_parity --dry-run

    # List fixtures
    python -m backend.scripts.measure_autoflip_parity --list

The runner is deliberately silent about which fixtures need future
phases to pass — every metric just gets a value (or ``None``) and the
results doc is the place where the targets live.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

from backend.services.autoflip_parity_fixtures import (
    FIXTURES,
    FixtureSpec,
    GroundTruth,
    get_fixture,
    list_fixture_names,
)
from backend.services.autoflip_parity_metrics import (
    extract_switch_times,
    score_fixture,
)


logger = logging.getLogger("autoflip_parity")


# ─────────────── Segmenter integration ───────────────────────

def _run_segmenter(spec: FixtureSpec) -> list:
    """Run ``build_reframe_segments`` against the fixture's inputs.

    Imported lazily so the rest of this module loads in a sandbox
    without numpy. The fixture's ``content_type_override`` /
    ``anime_subtype`` / ``music_subtype`` / ``game_type`` flow through
    the same metadata-injection contract that ``pipeline.py`` uses for
    real jobs; the test bench mirrors that contract exactly so the
    measured numbers are directly comparable to production.
    """
    from backend.services.reframe_segmenter import build_reframe_segments

    kwargs = spec.build()
    metadata: dict[str, str] = {}
    if spec.content_type_override:
        metadata["content_type_override"] = spec.content_type_override
    if spec.anime_subtype:
        metadata["anime_subtype"] = spec.anime_subtype
    if spec.music_subtype:
        metadata["music_subtype"] = spec.music_subtype
    if spec.game_type:
        metadata["game_type"] = spec.game_type

    # ``build_reframe_segments`` accepts a ``metadata`` kwarg only
    # when the caller wants to forward UI overrides. Older versions
    # didn't — in that case classify_content reads the override from
    # the global content_profile path. We try-fall here so the runner
    # works against both shapes without an explicit version check.
    try:
        return build_reframe_segments(metadata=metadata, **kwargs)
    except TypeError:
        return build_reframe_segments(**kwargs)


# ─────────────── Result extraction ────────────────────────────

def _segments_to_centers(
    segments: list,
    source_width: int,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Return per-frame crop-center % positions + segment spans.

    The reframe pipeline emits one ``ReframeSegment`` per visual
    decision. For metric scoring we want a per-frame x position
    (sampled at the same density as the fixture's dense face stream)
    so that ``max_acceleration`` / ``max_jerk`` see a uniform
    discrete-difference grid.

    Sampling rate matches ``measure_reframe_lag``: 30 fps. We sample
    every 1/30 s within each segment's [start, end) window.
    """
    if not segments:
        return [], []
    centers: list[float] = []
    spans: list[tuple[float, float]] = []
    fps = 30.0
    for seg in segments:
        start = float(getattr(seg, "start"))
        end = float(getattr(seg, "end"))
        spans.append((start, end))
        # subject_x is in source pixels in the segmenter; convert to %.
        sx_px = float(getattr(seg, "subject_x", source_width / 2.0))
        sx_pct = sx_px / max(source_width, 1) * 100.0
        n = max(1, int((end - start) * fps))
        for _ in range(n):
            centers.append(sx_pct)
    return centers, spans


def _spans_for_metric(segments: list) -> list[tuple[float, float]]:
    return [(float(s.start), float(s.end)) for s in segments]


# ─────────────── Per-fixture scoring ──────────────────────────

def score_one(
    spec: FixtureSpec,
    *,
    dry_run: bool = False,
) -> dict:
    """Score a single fixture and return a JSON-friendly dict."""

    base = {
        "name": spec.name,
        "description": spec.description,
        "content_type_override": spec.content_type_override,
        "anime_subtype": spec.anime_subtype,
        "music_subtype": spec.music_subtype,
        "game_type": spec.game_type,
        "video_duration": spec.video_duration,
        "metrics_requested": list(spec.metrics),
        "ground_truth_summary": _gt_summary(spec.ground_truth),
    }

    if dry_run:
        base["status"] = "dry_run"
        base["metrics"] = {}
        return base

    try:
        segments = _run_segmenter(spec)
    except ImportError as exc:
        # numpy / cv2 / etc. missing — this is the sandbox path.
        base["status"] = "skipped"
        base["error"] = f"segmenter import failed: {type(exc).__name__}: {exc}"
        base["metrics"] = {}
        return base
    except Exception as exc:
        base["status"] = "error"
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["metrics"] = {}
        return base

    crop_centers, spans = _segments_to_centers(segments, spec.source_width)
    actual_switches = extract_switch_times(segments)

    metrics_output = score_fixture(
        metrics_to_run=spec.metrics,
        expected_switches=spec.ground_truth.expected_switches,
        actual_switches=actual_switches,
        segment_spans=spans,
        crop_centers=crop_centers,
        required_regions_per_frame=spec.ground_truth.required_regions_per_frame,
        crop_width_pct=spec.crop_width_pct,
        beat_grid=spec.ground_truth.beat_grid,
        hud_zones=spec.ground_truth.hud_zones,
        face_y_in_crop_normalized=spec.ground_truth.face_y_in_crop_normalized,
    )

    base["status"] = "ok"
    base["segments"] = len(segments)
    base["actual_switches"] = len(actual_switches)
    base["metrics"] = metrics_output
    return base


def _gt_summary(gt: GroundTruth) -> dict:
    return {
        "expected_switches": len(gt.expected_switches),
        "beat_grid_size": len(gt.beat_grid),
        "required_region_frames": sum(
            1 for f in gt.required_regions_per_frame if f
        ),
        "hud_zones": len(gt.hud_zones),
        "game_key": gt.game_key,
    }


# ─────────────── Top-level runner ─────────────────────────────

def run_all(
    fixture_names: Optional[list[str]] = None,
    *,
    dry_run: bool = False,
) -> dict:
    names = fixture_names if fixture_names else list_fixture_names()
    rows = []
    for name in names:
        spec = get_fixture(name)
        rows.append(score_one(spec, dry_run=dry_run))
    return {
        "schema": "autoflip_parity_v2_results/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "fixture_count": len(rows),
        "results": rows,
    }


# ─────────────── CLI ─────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure_autoflip_parity",
        description=(
            "Score every Phase 9 fixture against its ground truth. "
            "See docs/autoflip_parity_v2_results.md for the target table."
        ),
    )
    p.add_argument(
        "--fixture",
        action="append",
        choices=sorted(FIXTURES.keys()),
        help="Score only the named fixture(s). Repeat to score several.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write the JSON result to this path (in addition to stdout).",
    )
    p.add_argument(
        "--phase",
        default=None,
        help=(
            "Optional phase tag stamped into the result payload "
            "(e.g. 'baseline', 'phase3', 'phase4')."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List fixture names + descriptions and exit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the segmenter call (sandbox / scaffolding test only).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logs.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list:
        for name in list_fixture_names():
            spec = get_fixture(name)
            print(f"{name:30s} {spec.description}")
        return 0

    payload = run_all(args.fixture, dry_run=args.dry_run)
    if args.phase:
        payload["phase"] = args.phase

    text = json.dumps(payload, indent=2)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        print(f"\nWrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
