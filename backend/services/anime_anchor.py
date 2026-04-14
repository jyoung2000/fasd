"""Phase 6 — Anime saliency anchor: picking the dramatic moment.

Anime reframing has a fundamentally different "right moment"
problem than live action. In a 2-second anime cut:

  - The camera holds for 1.5 s on a static-ish keyframe (the 2-on-3
    hold cadence), THEN explodes into 0.5 s of motion / impact /
    reveal.
  - The dramatic anchor isn't necessarily the speaker — it might
    be the attack landing, the spell effect, the reaction face, or
    the close-up tear.
  - Live-action speaker tracking would lock onto whoever's mouth
    moves, missing the anchor entirely.

This module ships a per-frame anime saliency scorer that combines
four signals:

  1. **Anime face detection** — the highest-confidence anime face
     in the frame (via :mod:`backend.services.anime_face_detector`).
     Highest priority because faces drive empathy and reaction
     shots are the single most-cut moment in anime editing.

  2. **Motion energy** — the absolute delta in mean pixel intensity
     against the previous frame, normalized to [0, 1]. Captures
     impact frames, attack swings, spell launches, panel zooms.

  3. **Contrast peak** — the standard deviation of pixel intensity
     normalized to [0, 1]. Captures dramatic lighting, rim-lit
     close-ups, silhouettes, light-beam reveals.

  4. **Color saturation peak** — the mean HSV saturation
     normalized to [0, 1]. Captures vivid effects, magic, energy
     attacks, the saturated-color flash that anime uses to mark
     "the moment".

Each signal is computed per frame and combined via a weighted
sum into a per-frame ``AnimeAnchor(x, y, score, source)`` where
``source`` is the dominant signal name. The reframe segmenter's
Stage 8 anime override picks the highest-scoring anchor across
each segment as the segment's ``subject_x``.

Production tier uses OpenCV / numpy lazily; sandbox tests
exercise the pure-Python scorers directly.

Feature flag: ``CLIPAI_ANIME_ANCHOR`` env var, default OFF.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_ANIME_ANCHOR = os.environ.get(
    "CLIPAI_ANIME_ANCHOR", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tuning ────────────────────

# Per-signal weights for the combined anchor score. Faces dominate
# (they're the single biggest drama predictor in anime), motion is
# second (action moments), contrast and saturation are tiebreakers.
WEIGHT_FACE = 0.50
WEIGHT_MOTION = 0.25
WEIGHT_CONTRAST = 0.15
WEIGHT_SATURATION = 0.10

# Subtype-specific multipliers — Phase 2 anime_subtype routes here.
# Action anime has more motion-driven moments; dialogue anime is
# more face-driven; slice-of-life sits between them.
SUBTYPE_WEIGHTS = {
    "action": {
        "face": 0.40, "motion": 0.40, "contrast": 0.10, "saturation": 0.10,
    },
    "dialogue": {
        "face": 0.65, "motion": 0.10, "contrast": 0.15, "saturation": 0.10,
    },
    "slice_of_life": {
        "face": 0.55, "motion": 0.20, "contrast": 0.15, "saturation": 0.10,
    },
}


# ──────────────────── Per-frame feature dataclass ────────────────────


@dataclass
class AnimeFrameFeatures:
    """Per-frame inputs for the anime anchor scorer.

    Attributes:
        timestamp: Frame timestamp in seconds.
        face_x_pct / face_y_pct / face_score: Position + confidence
            of the most-prominent anime face in the frame, in
            [0, 100] % coordinates. Set ``face_score=0`` when no
            face was detected.
        motion_energy: Absolute frame-to-frame intensity delta
            normalized to [0, 1]. Zero on the first frame of a
            shot or when no motion data is available.
        contrast: Pixel intensity standard deviation normalized
            to [0, 1].
        saturation: Mean HSV saturation normalized to [0, 1].
        motion_x_pct / contrast_x_pct: Optional spatial centroids
            for the motion / contrast signals when the caller has
            them. Default to 50 (frame center) so the anchor falls
            back to a centered crop when only the magnitude is
            known.
    """

    timestamp: float
    face_x_pct: float = 50.0
    face_y_pct: float = 50.0
    face_score: float = 0.0
    motion_energy: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    motion_x_pct: float = 50.0
    contrast_x_pct: float = 50.0


@dataclass
class AnimeAnchor:
    """Per-frame anime anchor decision.

    Attributes:
        timestamp: Frame timestamp in seconds.
        x_pct / y_pct: The chosen anchor position in [0, 100] %
            coordinates. The reframe segmenter uses ``x_pct`` as
            ``subject_x``.
        score: Combined anchor score in [0, 1]. Higher = stronger
            signal. The aggregator uses this to pick the most
            dramatic moment in a segment.
        source: Dominant signal name — one of ``"face"`` /
            ``"motion"`` / ``"contrast"`` / ``"saturation"`` /
            ``"fallback"``. Useful for telemetry and debug logs.
    """

    timestamp: float
    x_pct: float
    y_pct: float
    score: float
    source: str = "fallback"


# ──────────────────── Per-frame scorer ────────────────────


def _weights_for_subtype(anime_subtype: Optional[str]) -> dict:
    if anime_subtype and anime_subtype in SUBTYPE_WEIGHTS:
        return SUBTYPE_WEIGHTS[anime_subtype]
    return {
        "face": WEIGHT_FACE,
        "motion": WEIGHT_MOTION,
        "contrast": WEIGHT_CONTRAST,
        "saturation": WEIGHT_SATURATION,
    }


def score_anime_frame(
    features: AnimeFrameFeatures,
    *,
    anime_subtype: Optional[str] = None,
) -> AnimeAnchor:
    """Score a single frame's anime saliency and pick the anchor.

    Builds a weighted-sum score from the four signals, then
    returns an ``AnimeAnchor`` whose ``x_pct`` / ``y_pct`` come
    from the dominant signal's spatial centroid and whose
    ``source`` names that signal.

    Sub-type weights override the default mix when provided.
    """
    weights = _weights_for_subtype(anime_subtype)
    face_w = float(weights.get("face", WEIGHT_FACE))
    motion_w = float(weights.get("motion", WEIGHT_MOTION))
    contrast_w = float(weights.get("contrast", WEIGHT_CONTRAST))
    saturation_w = float(weights.get("saturation", WEIGHT_SATURATION))

    # Per-signal contribution to the combined score
    face_c = face_w * max(0.0, min(1.0, features.face_score))
    motion_c = motion_w * max(0.0, min(1.0, features.motion_energy))
    contrast_c = contrast_w * max(0.0, min(1.0, features.contrast))
    saturation_c = saturation_w * max(0.0, min(1.0, features.saturation))

    score = face_c + motion_c + contrast_c + saturation_c

    # Pick the dominant signal — its centroid wins the anchor
    candidates = [
        ("face", face_c, features.face_x_pct, features.face_y_pct),
        ("motion", motion_c, features.motion_x_pct, 50.0),
        ("contrast", contrast_c, features.contrast_x_pct, 50.0),
        # Saturation is a magnitude-only signal; fall to center.
        ("saturation", saturation_c, 50.0, 50.0),
    ]
    candidates.sort(key=lambda c: -c[1])
    top_source, top_contrib, top_x, top_y = candidates[0]
    if top_contrib <= 1e-9:
        # All signals zero → fall back to centered crop
        return AnimeAnchor(
            timestamp=features.timestamp,
            x_pct=50.0,
            y_pct=50.0,
            score=0.0,
            source="fallback",
        )

    return AnimeAnchor(
        timestamp=features.timestamp,
        x_pct=float(top_x),
        y_pct=float(top_y),
        score=max(0.0, min(1.0, score)),
        source=top_source,
    )


def score_anime_sequence(
    features_seq: Iterable[AnimeFrameFeatures],
    *,
    anime_subtype: Optional[str] = None,
) -> list[AnimeAnchor]:
    """Score every frame in a sequence and return per-frame anchors."""
    return [
        score_anime_frame(f, anime_subtype=anime_subtype)
        for f in features_seq
    ]


def snap_to_impact_frames(
    anchors: list[AnimeAnchor],
    features_seq: list,
    *,
    fps: float = 24.0,
    pre_impact_frames: int = 2,
    peak_threshold: float = 0.3,
) -> list[AnimeAnchor]:
    """Shift anchor hold start back by ``pre_impact_frames`` on motion
    peaks so anime action fights capture the wind-up before the blur.

    A local maximum is defined as a frame where ``motion_energy`` is
    greater than both neighbors on each side AND above ``peak_threshold``.

    The parallel ``features_seq`` carries motion_energy; ``anchors``
    and ``features_seq`` must be 1:1 by index. Returns a new list (the
    original anchors are not mutated except for peak-adjacent entries
    whose timestamps are shifted).
    """
    if len(anchors) < 5 or len(features_seq) != len(anchors):
        return anchors
    import dataclasses
    pre_shift = pre_impact_frames / max(fps, 1.0)
    energies = [
        float(getattr(f, "motion_energy", 0.0) or 0.0)
        for f in features_seq
    ]
    out = list(anchors)
    for i in range(2, len(out) - 2):
        if (
            energies[i] > peak_threshold
            and energies[i] > energies[i - 1]
            and energies[i] > energies[i - 2]
            and energies[i] > energies[i + 1]
            and energies[i] > energies[i + 2]
        ):
            prev = out[i - 1]
            new_ts = max(0.0, float(prev.timestamp) - pre_shift)
            out[i - 1] = dataclasses.replace(
                prev, timestamp=new_ts, source="pre_impact_hold",
            )
    return out


# ──────────────────── Per-segment aggregator ────────────────────


def best_anchor_in_window(
    anchors: list[AnimeAnchor],
    *,
    start: float,
    end: float,
    min_score: float = 0.0,
) -> Optional[AnimeAnchor]:
    """Return the highest-scoring anchor inside ``[start, end]``.

    Used by the reframe segmenter to pick the dramatic moment for
    each segment. Anchors with score below ``min_score`` are
    ignored (so very weak signals don't override the existing
    speaker-tracking decision).

    Returns None when no anchor is in range / above the floor.
    """
    best: Optional[AnimeAnchor] = None
    for a in anchors:
        if a.timestamp < start or a.timestamp > end:
            continue
        if a.score < min_score:
            continue
        if best is None or a.score > best.score:
            best = a
    return best


def aggregate_anchors_to_segment_x(
    anchors: list[AnimeAnchor],
    *,
    start: float,
    end: float,
    min_score: float = 0.30,
    fallback_x_pct: float = 50.0,
) -> tuple[float, str]:
    """Pick the segment's anime ``subject_x`` and source label.

    Calls :func:`best_anchor_in_window`; returns the winner's
    ``x_pct`` and ``source``. Falls back to ``fallback_x_pct`` /
    ``"fallback"`` when no anchor passes the score floor — the
    caller's existing speaker-tracking decision should win in
    that case.

    The default ``min_score=0.30`` matches the v2 spec's
    "moderate signal threshold" — below this, anime-anchor
    overriding the speaker tracker is a regression risk.
    """
    best = best_anchor_in_window(
        anchors, start=start, end=end, min_score=min_score,
    )
    if best is None:
        return (float(fallback_x_pct), "fallback")
    return (float(best.x_pct), best.source)


# ──────────────────── Pure-Python feature extractors ────────────────────


def motion_energy_from_intensity_means(
    intensity_means: list[float],
    *,
    norm_max: float = 30.0,
) -> list[float]:
    """Convert a sequence of per-frame mean intensities to motion energies.

    Returns a list of the same length where entry ``i`` is
    ``min(1.0, |intensity_means[i] - intensity_means[i-1]| / norm_max)``,
    with 0.0 for the first frame.

    ``norm_max`` defaults to 30 (out of 255) — a typical large
    intensity delta on an 8-bit frame normalizes to ~1.0.
    """
    if not intensity_means:
        return []
    out = [0.0]
    for i in range(1, len(intensity_means)):
        delta = abs(intensity_means[i] - intensity_means[i - 1])
        out.append(min(1.0, delta / max(norm_max, 1e-9)))
    return out


def normalize_intensity_stdev(stdev: float, *, norm_max: float = 64.0) -> float:
    """Convert an 8-bit intensity stdev to a normalized [0, 1] contrast score.

    A flat anime keyframe scores near 0; a high-contrast
    chiaroscuro frame scores near 1. Default ``norm_max=64``
    is half the max possible 8-bit stdev (128) so most frames
    land in [0, 1] without clipping.
    """
    return max(0.0, min(1.0, float(stdev) / max(norm_max, 1e-9)))


def normalize_saturation_mean(saturation_mean: float) -> float:
    """Convert an OpenCV HSV mean saturation (0-255) to [0, 1]."""
    return max(0.0, min(1.0, float(saturation_mean) / 255.0))
