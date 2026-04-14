"""Transcription coverage audit — verify Whisper didn't miss any dialogue.

Runs an independent Silero VAD pass on the audio file *after* Whisper has
produced its transcript segments, then diffs the two interval lists to
compute:

  * total_speech_seconds  — how much speech Silero thinks exists in the audio
  * covered_seconds       — how much of that speech overlaps Whisper segments
  * coverage_pct          — covered / total * 100
  * gaps                  — contiguous uncovered VAD intervals (>= min_gap_duration)

The audit is non-destructive and non-blocking — a failure to run it (e.g.
silero-vad not installed, ffmpeg missing) must never fail the pipeline.
Callers should swallow exceptions and log a warning.

This is Phase 1 of the "Close the Transcription Coverage Gap" project — pure
observability, no behavior change. Phase 2 uses the reported gaps to drive
targeted re-transcription.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CoverageGap:
    """A contiguous interval of speech (per Silero VAD) that is not covered
    by any transcript segment."""

    start: float
    end: float
    duration: float
    energy_rms: float  # RMS energy in this window; low RMS ≈ VAD false positive

    def to_dict(self) -> dict:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "duration": round(float(self.duration), 3),
            "energy_rms": round(float(self.energy_rms), 6),
        }


@dataclass
class CoverageReport:
    total_speech_seconds: float
    covered_seconds: float
    coverage_pct: float
    gaps: list[CoverageGap] = field(default_factory=list)
    vad_segment_count: int = 0
    transcript_segment_count: int = 0
    audio_duration: float = 0.0
    sample_rate: int = 16000
    vad_backend: str = "silero"          # "silero" or "unavailable"
    status: str = "ok"                   # "ok" | "skipped" | "error"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_speech_seconds": round(float(self.total_speech_seconds), 3),
            "covered_seconds": round(float(self.covered_seconds), 3),
            "coverage_pct": round(float(self.coverage_pct), 2),
            "gaps": [g.to_dict() for g in self.gaps],
            "vad_segment_count": int(self.vad_segment_count),
            "transcript_segment_count": int(self.transcript_segment_count),
            "audio_duration": round(float(self.audio_duration), 3),
            "sample_rate": int(self.sample_rate),
            "vad_backend": self.vad_backend,
            "status": self.status,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Audio loading (ffmpeg → raw PCM → numpy)
# ---------------------------------------------------------------------------


def _load_audio_mono_16k(audio_path: str) -> tuple["np.ndarray", int]:  # type: ignore[name-defined]
    """Decode ``audio_path`` to a float32 mono 16 kHz numpy array via ffmpeg.

    Used instead of soundfile/librosa to avoid adding a heavy audio dep —
    ffmpeg is already required everywhere else in the pipeline.
    """
    import numpy as np  # lazy import so module-level import never breaks tests

    target_sr = 16000
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", audio_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:300]}"
        )
    raw = proc.stdout
    if not raw:
        raise RuntimeError("ffmpeg produced no audio output")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, target_sr


# ---------------------------------------------------------------------------
# Silero VAD
# ---------------------------------------------------------------------------


def _run_silero_vad(
    audio: "np.ndarray",  # type: ignore[name-defined]
    sample_rate: int,
    onset: float = 0.3,
    offset: float = 0.2,
    min_speech_ms: int = 200,
    min_silence_ms: int = 200,
) -> list[tuple[float, float]]:
    """Run standalone Silero VAD on the given mono audio.

    Returns a list of ``(start_sec, end_sec)`` tuples of detected speech.
    Raises ``ModuleNotFoundError`` if silero-vad is not installed — the
    caller handles this and produces a "skipped" coverage report.

    Notes
    -----
    * We deliberately use the ``silero-vad`` PyPI package (not faster-whisper's
      internal silero copy) so we get an independent opinion about what is
      speech — a secondary signal that can disagree with the Whisper pipeline.
    * Silero expects 16 kHz mono float32 in [-1, 1]. ``_load_audio_mono_16k``
      already meets that contract.
    """
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore
    except ImportError as e:
        raise ModuleNotFoundError(
            "silero-vad not installed — coverage audit requires "
            "`pip install silero-vad` (optional dep)"
        ) from e

    import numpy as np
    import torch  # silero-vad depends on torch; if it's missing we'd already have raised

    if sample_rate != 16000:
        raise ValueError(f"Silero VAD requires 16 kHz audio (got {sample_rate})")

    model = load_silero_vad()
    audio_t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

    raw_ts = get_speech_timestamps(
        audio_t,
        model,
        sampling_rate=sample_rate,
        threshold=onset,
        neg_threshold=offset,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        return_seconds=True,
    )

    intervals: list[tuple[float, float]] = []
    for ts in raw_ts:
        # silero-vad returns dicts with 'start'/'end' in seconds when
        # return_seconds=True.
        try:
            start = float(ts["start"])
            end = float(ts["end"])
        except (KeyError, TypeError):
            continue
        if end > start:
            intervals.append((start, end))
    intervals.sort(key=lambda p: p[0])
    return intervals


# ---------------------------------------------------------------------------
# Pure interval math (unit-testable without silero/torch)
# ---------------------------------------------------------------------------


def _normalize_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort and merge overlapping/touching intervals."""
    cleaned = [
        (float(s), float(e))
        for s, e in intervals
        if e is not None and s is not None and float(e) > float(s)
    ]
    if not cleaned:
        return []
    cleaned.sort(key=lambda p: p[0])
    merged: list[tuple[float, float]] = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _segments_to_intervals(segments: Sequence[Any]) -> list[tuple[float, float]]:
    """Extract (start, end) tuples from a list of transcript segments.

    Accepts any object with .start/.end attrs (TranscriptSegment,
    CoverageGap, simple namespaces) *or* dicts.
    """
    out: list[tuple[float, float]] = []
    for seg in segments or []:
        if isinstance(seg, dict):
            s, e = seg.get("start"), seg.get("end")
        else:
            s, e = getattr(seg, "start", None), getattr(seg, "end", None)
        if s is None or e is None:
            continue
        s, e = float(s), float(e)
        if e > s:
            out.append((s, e))
    return out


def _overlap_seconds(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Length of overlap between two intervals."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def _subtract_intervals(
    base: tuple[float, float],
    subtract: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the parts of ``base`` not covered by any interval in
    ``subtract``. Both inputs must be sorted."""
    bs, be = base
    result: list[tuple[float, float]] = []
    cursor = bs
    for s, e in subtract:
        if e <= cursor:
            continue
        if s >= be:
            break
        s = max(s, cursor)
        e = min(e, be)
        if s > cursor:
            result.append((cursor, s))
        cursor = max(cursor, e)
        if cursor >= be:
            break
    if cursor < be:
        result.append((cursor, be))
    return result


def _compute_uncovered_intervals(
    vad_intervals: list[tuple[float, float]],
    transcript_intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """For each VAD speech interval, return the portions not covered by any
    transcript segment."""
    transcript_norm = _normalize_intervals(transcript_intervals)
    uncovered: list[tuple[float, float]] = []
    for v in vad_intervals:
        parts = _subtract_intervals(v, transcript_norm)
        uncovered.extend(parts)
    return uncovered


def _merge_close_gaps(
    gaps: list[tuple[float, float]],
    merge_threshold_sec: float,
) -> list[tuple[float, float]]:
    """Merge gaps whose edges are within ``merge_threshold_sec``."""
    if not gaps:
        return []
    merged: list[tuple[float, float]] = [gaps[0]]
    for s, e in gaps[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_threshold_sec:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _rms_energy(
    audio: Optional["np.ndarray"],  # type: ignore[name-defined]
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> float:
    """Compute RMS energy for the audio slice [start_sec, end_sec).

    Returns 0.0 when audio is None, out of range, or empty. Low RMS (<0.005)
    strongly suggests a VAD false positive.
    """
    if audio is None or len(audio) == 0 or end_sec <= start_sec:
        return 0.0
    import numpy as np

    i0 = max(0, int(start_sec * sample_rate))
    i1 = min(len(audio), int(end_sec * sample_rate))
    if i1 <= i0:
        return 0.0
    slice_ = audio[i0:i1]
    if slice_.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(slice_.astype(np.float32)))))


# ---------------------------------------------------------------------------
# Report builder (the testable entry point)
# ---------------------------------------------------------------------------


def build_coverage_report(
    vad_intervals: list[tuple[float, float]],
    transcript_segments: Sequence[Any],
    *,
    audio: Optional["np.ndarray"] = None,  # type: ignore[name-defined]
    sample_rate: int = 16000,
    audio_duration: float = 0.0,
    min_gap_duration: float = 0.8,
    merge_threshold_sec: float = 0.3,
    vad_backend: str = "silero",
) -> CoverageReport:
    """Pure-python report builder. Used by ``audit_transcription_coverage``
    *and* directly by unit tests that inject synthetic VAD intervals.

    Parameters
    ----------
    vad_intervals
        List of (start, end) tuples in seconds — already normalized.
    transcript_segments
        Sequence of TranscriptSegment-like objects with .start/.end or
        {"start": ..., "end": ...} dicts.
    audio
        Optional float32 mono audio array; used for RMS energy per gap.
    """
    vad_norm = _normalize_intervals(vad_intervals)
    transcript_intervals = _segments_to_intervals(transcript_segments)

    total_speech = sum(e - s for s, e in vad_norm)
    uncovered = _compute_uncovered_intervals(vad_norm, transcript_intervals)
    total_uncovered = sum(e - s for s, e in uncovered)
    covered = max(0.0, total_speech - total_uncovered)

    if total_speech < 1e-3:
        # No detectable speech → nothing could be missed. Coverage is 100% by
        # convention so silence clips don't flag as failures.
        coverage_pct = 100.0
    else:
        coverage_pct = (covered / total_speech) * 100.0

    # Gap selection: only uncovered intervals >= min_gap_duration. Merge
    # first, then threshold — two 0.5s uncovered intervals 0.2s apart
    # are the same missed utterance.
    merged_uncovered = _merge_close_gaps(uncovered, merge_threshold_sec)
    gaps: list[CoverageGap] = []
    for s, e in merged_uncovered:
        dur = e - s
        if dur < min_gap_duration:
            continue
        rms = _rms_energy(audio, sample_rate, s, e)
        gaps.append(
            CoverageGap(
                start=float(s),
                end=float(e),
                duration=float(dur),
                energy_rms=float(rms),
            )
        )

    return CoverageReport(
        total_speech_seconds=float(total_speech),
        covered_seconds=float(covered),
        coverage_pct=float(coverage_pct),
        gaps=gaps,
        vad_segment_count=len(vad_norm),
        transcript_segment_count=len(transcript_intervals),
        audio_duration=float(audio_duration),
        sample_rate=int(sample_rate),
        vad_backend=vad_backend,
        status="ok",
    )


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------


async def audit_transcription_coverage(
    audio_path: str,
    segments: Sequence[Any],
    *,
    min_gap_duration: float = 0.8,
    vad_onset: float = 0.3,
    vad_offset: float = 0.2,
    merge_threshold_sec: float = 0.3,
) -> CoverageReport:
    """Run Silero VAD on ``audio_path`` and diff against transcript segments.

    Parameters
    ----------
    audio_path
        Path to the audio file used for transcription. Preferably the
        preprocessed 16 kHz mono wav that Whisper actually saw.
    segments
        The TranscriptSegment list produced by Whisper (or any iterable of
        objects/dicts with ``start``/``end`` fields).
    min_gap_duration
        Smallest uncovered interval (seconds) that is reported as a gap.
        Shorter gaps are ignored — they're almost always breath/boundary
        jitter, not missed dialogue.
    vad_onset, vad_offset
        Silero VAD probability thresholds. Defaults (0.3 / 0.2) are a little
        more permissive than Silero's stock (0.5 / 0.35) so we notice
        soft-spoken Japanese and whispered backchannels.

    Returns
    -------
    CoverageReport
        Always returns a report. On any failure (missing dep, bad audio,
        etc.) returns a report with ``status="skipped"`` or ``"error"`` and
        coverage_pct=0 so callers can distinguish "100% because nothing to
        check" from "we never ran".
    """
    if not audio_path or not os.path.exists(audio_path):
        logger.warning("coverage_audit: audio file missing: %r", audio_path)
        return CoverageReport(
            total_speech_seconds=0.0,
            covered_seconds=0.0,
            coverage_pct=0.0,
            vad_segment_count=0,
            transcript_segment_count=len(_segments_to_intervals(segments)),
            vad_backend="unavailable",
            status="error",
            error=f"audio not found: {audio_path}",
        )

    def _work() -> CoverageReport:
        # Decode audio first — if ffmpeg or numpy is missing we return a
        # clean "skipped" report rather than raising into the pipeline.
        try:
            audio, sr = _load_audio_mono_16k(audio_path)
        except Exception as e:
            logger.warning("coverage_audit: audio decode failed: %s", e)
            return CoverageReport(
                total_speech_seconds=0.0,
                covered_seconds=0.0,
                coverage_pct=0.0,
                transcript_segment_count=len(_segments_to_intervals(segments)),
                vad_backend="unavailable",
                status="error",
                error=f"audio decode failed: {e}",
            )

        audio_duration = len(audio) / float(sr) if sr else 0.0

        try:
            vad_intervals = _run_silero_vad(
                audio, sr, onset=vad_onset, offset=vad_offset
            )
            backend = "silero"
        except ModuleNotFoundError as e:
            logger.warning("coverage_audit: %s", e)
            return CoverageReport(
                total_speech_seconds=0.0,
                covered_seconds=0.0,
                coverage_pct=0.0,
                transcript_segment_count=len(_segments_to_intervals(segments)),
                audio_duration=audio_duration,
                sample_rate=sr,
                vad_backend="unavailable",
                status="skipped",
                error="silero-vad not installed",
            )
        except Exception as e:
            logger.warning("coverage_audit: silero VAD failed: %s", e)
            return CoverageReport(
                total_speech_seconds=0.0,
                covered_seconds=0.0,
                coverage_pct=0.0,
                transcript_segment_count=len(_segments_to_intervals(segments)),
                audio_duration=audio_duration,
                sample_rate=sr,
                vad_backend="unavailable",
                status="error",
                error=f"silero VAD error: {e}",
            )

        return build_coverage_report(
            vad_intervals,
            segments,
            audio=audio,
            sample_rate=sr,
            audio_duration=audio_duration,
            min_gap_duration=min_gap_duration,
            merge_threshold_sec=merge_threshold_sec,
            vad_backend=backend,
        )

    # Silero VAD on CPU for a few minutes of audio is ~100-400ms — cheap,
    # but still blocking. Push to a worker thread so we don't stall the
    # pipeline event loop.
    return await asyncio.to_thread(_work)


__all__ = [
    "CoverageGap",
    "CoverageReport",
    "audit_transcription_coverage",
    "build_coverage_report",
]
