"""Phase 3 — classification_hint populated for low-confidence routing.

When the auto-router doesn't pick gameplay BUT the HUD or
stylization signals are close, we surface a banner so the user
can confirm the content type. Same for anime: if
``ANIME_MODE_DETECTED`` fires AFTER classification was locked to
live-action, suggest ``"anime"``.

These tests pin the ``build_classification_hint`` decision rules
without standing up the full pipeline.
"""

from __future__ import annotations

import pytest

from backend.services.face_detector import build_classification_hint


# ──────────────────── Hint suppression cases ────────────────────


def test_hint_empty_when_decision_was_gameplay():
    """If the auto-router already picked gameplay, no banner."""
    h = build_classification_hint(
        "gameplay",
        {"hud": 0.7, "stylization": 0.6, "face_ratio": 0.1, "crosshair": 0.2},
    )
    assert h == {}


def test_hint_empty_when_user_overrode():
    """User explicit choice always wins — no banner."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.8, "stylization": 0.7, "face_ratio": 0.1, "crosshair": 0.0},
        user_overridden=True,
    )
    assert h == {}


def test_hint_empty_when_signals_below_threshold():
    """A normal podcast clip with no HUD or cartoon signals → no hint."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.05, "stylization": 0.10, "face_ratio": 0.85, "crosshair": 0.0},
    )
    assert h == {}


# ──────────────────── Gaming hint cases ────────────────────


def test_hint_gaming_when_high_hud():
    """High HUD score but face-route picked → suggest gaming."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.72, "stylization": 0.30, "face_ratio": 0.42, "crosshair": 0.10},
    )
    assert h["suggested_content_type"] == "gaming"
    assert h["suggested_game_type"] == "generic_fps"
    assert h["reason"] == "high_hud_score"
    assert h["scores"]["hud"] == 0.72
    assert h["scores"]["face_ratio"] == 0.42


def test_hint_gaming_when_high_stylization():
    """Cartoon-shooter case where HUD is borderline but
    stylization score is high → still suggest gaming."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.30, "stylization": 0.65, "face_ratio": 0.45, "crosshair": 0.10},
    )
    assert h["suggested_content_type"] == "gaming"
    assert h["reason"] == "high_stylization"


def test_hint_gaming_uses_safe_default_game_type():
    """We never guess specific games — always suggest generic_fps
    as the safe default (per the spec note)."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.85, "stylization": 0.20, "face_ratio": 0.20, "crosshair": 0.0},
    )
    assert h["suggested_game_type"] == "generic_fps"


# ──────────────────── Anime hint cases ────────────────────


def test_hint_anime_when_anime_mode_detected_after_face_route():
    """If ANIME_MODE_DETECTED fires AFTER the auto-router already
    picked face-route (or unknown), suggest anime."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.10, "stylization": 0.60, "face_ratio": 0.40, "crosshair": 0.0},
        anime_mode_detected=True,
    )
    assert h["suggested_content_type"] == "anime"
    assert h["reason"] == "anime_mode_detected"


def test_anime_hint_takes_precedence_over_gaming_hint():
    """When BOTH anime mode and gaming HUD signals are present,
    the anime branch is more specific and wins."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.80, "stylization": 0.70, "face_ratio": 0.30, "crosshair": 0.0},
        anime_mode_detected=True,
    )
    assert h["suggested_content_type"] == "anime"


def test_anime_hint_suppressed_when_classification_was_gameplay():
    """If we already routed to gameplay, no anime hint either —
    the user can override later if needed."""
    h = build_classification_hint(
        "gameplay",
        {"hud": 0.80, "stylization": 0.70, "face_ratio": 0.30, "crosshair": 0.0},
        anime_mode_detected=True,
    )
    assert h == {}


# ──────────────────── Score plumbing tests ────────────────────


def test_hint_scores_block_is_rounded():
    """The score block should round to 2 decimal places for display."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.7234, "stylization": 0.4567, "face_ratio": 0.4123, "crosshair": 0.1},
    )
    assert h["scores"] == {
        "hud": 0.72,
        "stylization": 0.46,
        "face_ratio": 0.41,
        "crosshair": 0.1,
    }


def test_hint_handles_missing_score_keys():
    """An incomplete scores dict shouldn't crash — defaults to 0.0."""
    h = build_classification_hint(
        "not_gameplay",
        {"hud": 0.7},  # only hud
    )
    assert h["suggested_content_type"] == "gaming"
    assert h["scores"]["face_ratio"] == 0.0


def test_hint_handles_none_score_values():
    """``None`` score values from a stale cache shouldn't crash."""
    h = build_classification_hint(
        "not_gameplay",
        {
            "hud": None, "stylization": 0.6,
            "face_ratio": None, "crosshair": None,
        },
    )
    assert h["suggested_content_type"] == "gaming"


# ──────────────────── classify_gameplay_content_with_scores ────────────────────


def test_classify_with_scores_returns_decision_and_dict():
    """The wrapper must return a (decision, scores) tuple even on
    a degenerate input."""
    pytest.importorskip("numpy")
    from backend.services.face_detector import (
        classify_gameplay_content_with_scores,
    )
    decision, scores = classify_gameplay_content_with_scores(
        dense_face_data=[],
        total_frames=0,
        sample_frame_paths=[],
    )
    assert decision in ("gameplay", "not_gameplay", "unknown")
    # Scores dict should always have the four keys
    assert set(scores) >= {"face_ratio", "crosshair", "hud", "stylization"}


# ──────────────────── Job model field tests ────────────────────


def test_job_model_has_classification_hint_field():
    """The Job model must define ``classification_hint: dict = {}``
    so ``database.update_job_status(classification_hint=...)`` lands
    on a real attribute."""
    from backend.models import JobResult
    # Pydantic field exists with a default
    fields = JobResult.model_fields if hasattr(JobResult, "model_fields") else {}
    assert "classification_hint" in fields, (
        "JobResult must expose classification_hint for Phase 3"
    )
