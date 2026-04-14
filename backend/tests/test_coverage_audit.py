"""Unit tests for backend.services.coverage_audit.

Phase 1 of the "Close the Transcription Coverage Gap" project. These tests
exercise the pure-python interval math in ``build_coverage_report`` so they
do NOT require silero-vad, torch, ffmpeg, or any audio files. The Silero
pass is tested indirectly: callers can inject synthetic VAD intervals.

The one "real audio" test (``test_audit_with_missing_audio_returns_error``)
only exercises the error path when the file doesn't exist — still no
external deps.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.services.coverage_audit import (
    CoverageGap,
    CoverageReport,
    _compute_uncovered_intervals,
    _merge_close_gaps,
    _normalize_intervals,
    _segments_to_intervals,
    _subtract_intervals,
    audit_transcription_coverage,
    build_coverage_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(start: float, end: float, text: str = "x") -> SimpleNamespace:
    """A minimal stand-in for TranscriptSegment."""
    return SimpleNamespace(start=float(start), end=float(end), text=text)


# ---------------------------------------------------------------------------
# Pure interval math
# ---------------------------------------------------------------------------


def test_normalize_intervals_merges_overlapping():
    merged = _normalize_intervals([(5, 8), (0, 3), (2, 6), (10, 11)])
    assert merged == [(0.0, 8.0), (10.0, 11.0)]


def test_normalize_intervals_drops_zero_width():
    merged = _normalize_intervals([(1, 1), (2, 3), (4, 4)])
    assert merged == [(2.0, 3.0)]


def test_subtract_intervals_basic():
    # base [0, 10], subtract [2, 4] and [6, 7] → [(0,2), (4,6), (7,10)]
    parts = _subtract_intervals((0, 10), [(2, 4), (6, 7)])
    assert parts == [(0.0, 2.0), (4.0, 6.0), (7.0, 10.0)]


def test_subtract_intervals_covered_entirely():
    assert _subtract_intervals((0, 10), [(0, 10)]) == []
    assert _subtract_intervals((2, 5), [(0, 100)]) == []


def test_subtract_intervals_no_overlap():
    assert _subtract_intervals((0, 5), [(10, 20)]) == [(0.0, 5.0)]


def test_segments_to_intervals_accepts_objects_and_dicts():
    segs = [
        _seg(0, 1),
        {"start": 2, "end": 3},
        SimpleNamespace(start=4, end=4),     # zero width — dropped
        SimpleNamespace(start=None, end=5),  # missing start — dropped
    ]
    assert _segments_to_intervals(segs) == [(0.0, 1.0), (2.0, 3.0)]


def test_compute_uncovered_intervals_returns_holes_per_vad_interval():
    vad = [(0, 10), (20, 30)]
    transcript = [(0, 8), (20, 25)]
    uncovered = _compute_uncovered_intervals(vad, transcript)
    assert uncovered == [(8.0, 10.0), (25.0, 30.0)]


def test_merge_close_gaps_merges_within_threshold():
    gaps = [(0.0, 1.0), (1.2, 2.0), (5.0, 6.0)]
    merged = _merge_close_gaps(gaps, merge_threshold_sec=0.3)
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_merge_close_gaps_empty():
    assert _merge_close_gaps([], 0.3) == []


# ---------------------------------------------------------------------------
# build_coverage_report — the acceptance-criteria tests
# ---------------------------------------------------------------------------


def test_three_speech_blocks_third_uncovered():
    """Synthetic case from the spec:

    Audio has 3 speech blocks (0–10s, 20–30s, 40–50s) and transcript covers
    only the first two → coverage ≈ 66.7%, 1 gap at 40–50s.
    """
    vad = [(0, 10), (20, 30), (40, 50)]
    transcript = [_seg(0, 10), _seg(20, 30)]

    report = build_coverage_report(vad, transcript, audio_duration=60.0)

    assert report.total_speech_seconds == pytest.approx(30.0)
    assert report.covered_seconds == pytest.approx(20.0)
    assert report.coverage_pct == pytest.approx(66.666, rel=1e-3)
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.start == pytest.approx(40.0)
    assert gap.end == pytest.approx(50.0)
    assert gap.duration == pytest.approx(10.0)
    assert report.vad_segment_count == 3
    assert report.transcript_segment_count == 2


def test_full_coverage_produces_no_gaps():
    """Transcript covers all VAD intervals → coverage 100%, 0 gaps."""
    vad = [(0, 10), (15, 20)]
    transcript = [_seg(0, 10), _seg(15, 20)]

    report = build_coverage_report(vad, transcript)

    assert report.coverage_pct == pytest.approx(100.0)
    assert report.gaps == []
    # Must also satisfy the ≥95% clean-clip acceptance criterion
    assert report.coverage_pct >= 95.0


def test_partial_coverage_within_single_vad_interval():
    """Transcript partially covers a VAD interval — should still report the
    uncovered tail as a gap."""
    vad = [(0, 10)]
    transcript = [_seg(0, 3), _seg(4, 6)]  # 5s covered, 5s uncovered in holes

    report = build_coverage_report(vad, transcript)

    assert report.total_speech_seconds == pytest.approx(10.0)
    assert report.covered_seconds == pytest.approx(5.0)
    assert report.coverage_pct == pytest.approx(50.0)
    # Two gaps > 0.8s: (3,4) is only 1s — reported. (6,10) is 4s — reported.
    assert len(report.gaps) == 2
    assert report.gaps[0].start == pytest.approx(3.0)
    assert report.gaps[0].end == pytest.approx(4.0)
    assert report.gaps[1].start == pytest.approx(6.0)
    assert report.gaps[1].end == pytest.approx(10.0)


def test_silence_only_audio_returns_100pct_no_gaps():
    """If VAD reports zero speech, coverage is 100% by convention — there
    is nothing to miss."""
    report = build_coverage_report([], [], audio_duration=60.0)
    assert report.total_speech_seconds == 0.0
    assert report.coverage_pct == pytest.approx(100.0)
    assert report.gaps == []
    assert report.vad_segment_count == 0


def test_gap_merging_two_adjacent_uncovered_become_one():
    """Two VAD intervals 0.2s apart that are both uncovered should merge
    into a single reported gap (merge_threshold_sec default is 0.3s)."""
    vad = [(0, 2), (2.2, 5)]
    transcript: list[SimpleNamespace] = []

    report = build_coverage_report(vad, transcript)

    # 2s + 2.8s = 4.8s of "speech", all uncovered. Merged into one gap
    # spanning 0 → 5.
    assert report.total_speech_seconds == pytest.approx(4.8)
    assert len(report.gaps) == 1
    assert report.gaps[0].start == pytest.approx(0.0)
    assert report.gaps[0].end == pytest.approx(5.0)


def test_gap_shorter_than_min_duration_is_filtered():
    """A 0.5s uncovered interval should NOT be reported under the default
    min_gap_duration of 0.8s — it's breath/boundary jitter, not a missed
    utterance."""
    vad = [(0, 5)]
    transcript = [_seg(0, 1.5), _seg(2.0, 5.0)]  # 0.5s gap at 1.5-2.0

    report = build_coverage_report(vad, transcript)

    assert report.gaps == []
    # Coverage should still reflect the missed 0.5s
    assert report.coverage_pct == pytest.approx(90.0, rel=1e-3)


def test_short_gap_appears_when_min_gap_duration_lowered():
    """Same scenario as above but with min_gap_duration=0.3 should surface
    the gap — proves the knob works."""
    vad = [(0, 5)]
    transcript = [_seg(0, 1.5), _seg(2.0, 5.0)]

    report = build_coverage_report(vad, transcript, min_gap_duration=0.3)

    assert len(report.gaps) == 1
    assert report.gaps[0].duration == pytest.approx(0.5)


def test_low_energy_gap_is_still_reported_but_flagged():
    """Low-RMS "gaps" (VAD false positives in quiet audio) are still
    reported — with energy_rms preserved so the caller/UI can decide to
    ignore them. This is the policy from the spec.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy required for RMS energy test")

    # Build a 10s mono 16k "audio" array of essentially silence
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    # Put a quiet 2s region at [3, 5] with an RMS well below 0.005
    audio[sr * 3 : sr * 5] = np.random.uniform(-1e-4, 1e-4, size=sr * 2)
    # And a loud 2s region at [6, 8] with clear signal (RMS ≈ 0.5)
    audio[sr * 6 : sr * 8] = np.random.uniform(-0.7, 0.7, size=sr * 2)

    vad = [(3, 5), (6, 8)]  # Silero reports both as "speech"
    transcript: list[SimpleNamespace] = []     # Nothing transcribed

    report = build_coverage_report(
        vad, transcript, audio=audio, sample_rate=sr
    )

    assert len(report.gaps) == 2
    quiet, loud = report.gaps
    # The quiet gap has tiny energy
    assert quiet.energy_rms < 0.005
    # The loud gap has substantial energy
    assert loud.energy_rms > 0.1
    # Both are still present — the user/UI filters on RMS, not us
    assert quiet.duration == pytest.approx(2.0)
    assert loud.duration == pytest.approx(2.0)


def test_report_to_dict_roundtrips_all_fields():
    report = build_coverage_report(
        [(0, 10), (15, 20)], [_seg(0, 10)], audio_duration=25.0
    )
    d = report.to_dict()
    # Standard shape: every field the frontend / persistence layer expects
    for key in (
        "total_speech_seconds",
        "covered_seconds",
        "coverage_pct",
        "gaps",
        "vad_segment_count",
        "transcript_segment_count",
        "audio_duration",
        "sample_rate",
        "vad_backend",
        "status",
    ):
        assert key in d
    assert isinstance(d["gaps"], list)
    assert len(d["gaps"]) == 1
    assert set(d["gaps"][0].keys()) == {"start", "end", "duration", "energy_rms"}


# ---------------------------------------------------------------------------
# The async entry point — error paths (no real audio needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_with_missing_audio_returns_error():
    report = await audit_transcription_coverage(
        audio_path="/tmp/definitely-not-a-real-file-xyz-123.wav",
        segments=[_seg(0, 5)],
    )
    assert isinstance(report, CoverageReport)
    assert report.status == "error"
    assert report.vad_backend == "unavailable"
    assert report.coverage_pct == 0.0
    # Transcript seg count still surfaces so the UI can say
    # "audit unavailable, you have N segments"
    assert report.transcript_segment_count == 1


@pytest.mark.asyncio
async def test_audit_with_empty_path_returns_error():
    report = await audit_transcription_coverage(audio_path="", segments=[])
    assert report.status == "error"
    assert "not found" in (report.error or "").lower()


# ---------------------------------------------------------------------------
# Coverage gap dataclass
# ---------------------------------------------------------------------------


def test_coverage_gap_to_dict_rounds_floats():
    gap = CoverageGap(start=1.23456789, end=2.98765, duration=1.75308, energy_rms=0.123456)
    d = gap.to_dict()
    assert d["start"] == 1.235
    assert d["end"] == 2.988
    assert d["duration"] == 1.753
    assert d["energy_rms"] == 0.123456
