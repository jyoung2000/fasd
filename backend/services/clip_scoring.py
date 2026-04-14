"""Composite clip scoring — Phase 4 of the OpusClip parity gap.

The LLM no longer returns a single ``viral_score``. Instead it returns
four axis scores (hook / flow / value / trend) and we combine them into
a composite using genre-aware weights. This module owns the math.

The composite is deterministic and unit-tested. ``clip_verifier`` and
the existing hook-strength / retention adjustments still apply on top of
the composite, so the LLM's per-axis judgement remains the dominant
signal but local heuristics can nudge it.

Backward compatibility: when all four axis scores are zero (legacy
clips that never went through the new prompt) we leave ``viral_score``
untouched.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from backend.models import ClipCandidate
from backend.services.content_classifier import ClipContentType

logger = logging.getLogger(__name__)


# Default 4-axis weights when no genre is known. Hook + Value are the
# two strongest predictors of completion / re-watch on short-form
# platforms, with Flow and Trend as supporting signals.
GENRE_WEIGHTS_DEFAULT: dict[str, float] = {
    "hook": 0.30,
    "flow": 0.25,
    "value": 0.30,
    "trend": 0.15,
}


# Per-genre weight overrides. Each row must sum to 1.0 (we assert in
# unit tests). Tuned for the OpusClip parity gap — gameplay / sports
# weight Hook hardest because the first few frames of a clutch play
# decide whether a viewer keeps watching, while talking-head weights
# Value highest because the substance of what's said is what gets
# shared.
GENRE_WEIGHTS: dict[ClipContentType, dict[str, float]] = {
    ClipContentType.TALKING_HEAD: {
        "hook": 0.30, "flow": 0.25, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.MULTI_SPEAKER_PANEL: {
        "hook": 0.25, "flow": 0.30, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.CINEMATIC_DIALOGUE: {
        "hook": 0.25, "flow": 0.35, "value": 0.30, "trend": 0.10,
    },
    ClipContentType.GAMEPLAY: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.GAMEPLAY_MOBA: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.GAMEPLAY_TPS: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.GAMEPLAY_RACING: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.STREAM: {
        "hook": 0.35, "flow": 0.20, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.SPORTS: {
        "hook": 0.45, "flow": 0.10, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.SPORTS_BASKETBALL: {
        "hook": 0.45, "flow": 0.10, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.SPORTS_RACING: {
        "hook": 0.45, "flow": 0.10, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.MUSIC_VIDEO: {
        "hook": 0.40, "flow": 0.20, "value": 0.15, "trend": 0.25,
    },
    ClipContentType.ANIMATION: {
        "hook": 0.35, "flow": 0.25, "value": 0.25, "trend": 0.15,
    },
    ClipContentType.ANIMATION_DIALOGUE: {
        "hook": 0.30, "flow": 0.30, "value": 0.25, "trend": 0.15,
    },
    ClipContentType.GENERIC: GENRE_WEIGHTS_DEFAULT,
}


def _flag_enabled(name: str, default: bool) -> bool:
    """Read a feature flag from the environment.

    Truthy values: 1 / true / yes / on (case-insensitive). Anything else
    is falsy. Allows operators to flip Phase 1/2 behavior without code
    changes.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def four_axis_scoring_enabled() -> bool:
    """Phase 1 feature flag — defaults on per the gap-close brief."""
    return _flag_enabled("USE_FOUR_AXIS_SCORING", True)


def get_weights(content_type: Optional[ClipContentType]) -> dict[str, float]:
    """Return the axis weights to use for a given content type.

    Always returns a dict with keys hook/flow/value/trend that sum to
    ~1.0. Falls back to GENRE_WEIGHTS_DEFAULT for unknown types.
    """
    if content_type is None:
        return GENRE_WEIGHTS_DEFAULT
    return GENRE_WEIGHTS.get(content_type, GENRE_WEIGHTS_DEFAULT)


def composite_score(
    clip: ClipCandidate,
    content_type: Optional[ClipContentType] = None,
) -> int:
    """Compute the composite viral_score from the four axes.

    Returns an int 1-100. When all four axes are 0 (legacy clip from a
    pre-Phase-1 prompt), returns the existing viral_score unchanged so
    we never silently zero out a clip's score. When at least one axis
    is populated, the composite is the genre-weighted average of all
    four axes.
    """
    axes = (
        int(clip.hook_score),
        int(clip.flow_score),
        int(clip.value_score),
        int(clip.trend_score),
    )
    if all(a == 0 for a in axes):
        # Legacy path — nothing to compose. Keep the LLM-supplied score.
        return max(1, min(100, int(clip.viral_score) if clip.viral_score else 50))

    weights = get_weights(content_type)
    raw = (
        axes[0] * weights["hook"]
        + axes[1] * weights["flow"]
        + axes[2] * weights["value"]
        + axes[3] * weights["trend"]
    )
    return max(1, min(100, int(round(raw))))


def finalize_clip_scores(
    clips: list[ClipCandidate],
    content_type: Optional[ClipContentType] = None,
) -> list[ClipCandidate]:
    """Compute composite viral_score for every clip in-place.

    Also populates ``clip.score_diagnostics`` with the weights used and
    the per-axis breakdown so the job result JSON can be inspected
    after the fact.
    """
    if not four_axis_scoring_enabled():
        return clips

    weights = get_weights(content_type)
    for clip in clips:
        new_score = composite_score(clip, content_type)
        prev_score = clip.viral_score
        clip.viral_score = new_score
        diag = dict(clip.score_diagnostics or {})
        diag.update({
            "axis_scores": {
                "hook": int(clip.hook_score),
                "flow": int(clip.flow_score),
                "value": int(clip.value_score),
                "trend": int(clip.trend_score),
            },
            "axis_reasons": {
                "hook": clip.hook_reason,
                "flow": clip.flow_reason,
                "value": clip.value_reason,
                "trend": clip.trend_reason,
            },
            "weights": dict(weights),
            "content_type": content_type.value if content_type else None,
            "composite_before": int(prev_score),
            "composite_after": int(new_score),
        })
        clip.score_diagnostics = diag
    return clips


def fill_axes_from_legacy(clip: ClipCandidate) -> None:
    """Backfill axis scores from a legacy ``viral_score`` field.

    Used by the focus-query path (``clip_focus``) and by salvage paths
    where the LLM didn't return the four axes. We don't know how to
    decompose the score, so we splat it across all four axes — this
    keeps downstream UI code that reads ``hook_score`` etc. happy
    without overstating the signal.
    """
    score = max(1, min(100, int(clip.viral_score or 0)))
    if clip.hook_score == 0:
        clip.hook_score = score
        clip.hook_reason = clip.hook_reason or "derived from legacy viral_score"
    if clip.flow_score == 0:
        clip.flow_score = score
        clip.flow_reason = clip.flow_reason or "derived from legacy viral_score"
    if clip.value_score == 0:
        clip.value_score = score
        clip.value_reason = clip.value_reason or "derived from legacy viral_score"
    if clip.trend_score == 0:
        # Trend defaults to 50 (neutral) when no real signal exists,
        # not the legacy viral_score — overstating trend would let
        # neutral content beat genuinely on-trend clips.
        clip.trend_score = 50
        clip.trend_reason = clip.trend_reason or "no trend signal — neutral default"
