"""Canonical string mapping for UI-selected content types.

The reframe pipeline has multiple layers of content-type bookkeeping:

- The upload UI dropdown sends a string token (``content_type_override``).
- ``content_classifier.classify_content`` picks a ``ContentType`` enum value.
- ``classify_clip`` maps that to a ``ClipContentType`` for solver tuning.

Historically these layers had mismatched string spellings — the UI
would send ``"gameplay"`` but the enum value is ``"gaming"``; the UI
would send ``"movie"`` but the enum value is ``"narrative"``; only
``"podcast"`` matched by coincidence. The result was that the
classifier's user-override branch silently never fired for two of the
three existing dropdown values, and new UI values (``debate``,
``vlog``, ``anime``, ``music_video``, the gameplay variants, etc.)
would all hit the same dead code path.

This module is the single source of truth for the UI → enum mapping
so:

1. Every UI token is normalized to a ``ContentType`` enum + a bundle
   of editorial flags (``is_multi_speaker_panel``, ``is_animated``).
2. Legacy UI aliases (``"gameplay"`` / ``"movie"``) keep working
   forever.
3. Forward-compat tokens for Phase 2 (``debate``, ``vlog``, ``anime``,
   ``music_video``, ``gameplay_moba``, ``gameplay_tps``,
   ``gameplay_racing``, ``stream``, ``sports``) all normalize today so
   we don't have to touch ``content_classifier.py`` again when the
   dropdown grows.
4. Unknown / empty / ``"auto"`` tokens return ``None`` so the caller
   falls through to heuristic classification.

``is_gameplay_override`` is a convenience for the pipeline's early
gameplay fast-path, which skips face tracking entirely. Note that
``"stream"`` is deliberately NOT a gameplay fast-path candidate — a
stream has a facecam which requires the face pipeline — so it
normalizes to ``ContentType.GAMING`` but is_gameplay_override is
False for it. Phase 7 handles the stream layout routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from backend.services.content_type_config import ContentType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedContentType:
    """Result of normalizing a UI content-type string.

    Attributes:
        content_type: The canonical ``ContentType`` enum value.
        is_multi_speaker_panel: True when the UI explicitly picked a
            debate / panel format. Drives ``ClipContentType.MULTI_SPEAKER_PANEL``
            routing in ``classify_clip``.
        is_animated: True when the UI picked anime/cartoon. Routes to
            ``ClipContentType.ANIMATION`` / ``ANIMATION_DIALOGUE`` downstream.
        anime_subtype: "action" | "dialogue" | "slice_of_life" | None.
            Reserved for the Phase 2 UI sub-dropdown.
        music_subtype: "performance" | "narrative" | "lyric" | None.
            Reserved for the Phase 2 UI sub-dropdown.
        is_gameplay_fastpath: True when the pipeline should skip face
            tracking entirely (pure gameplay). False for ``stream``,
            which is gaming-layout but still needs the face pipeline.
        raw: The original UI token (lowercased + stripped). Preserved
            for logging / telemetry.
    """

    content_type: ContentType
    is_multi_speaker_panel: bool = False
    is_animated: bool = False
    anime_subtype: Optional[str] = None
    music_subtype: Optional[str] = None
    is_gameplay_fastpath: bool = False
    raw: str = ""


# Every dropdown value in Upload.jsx MUST appear here. The legacy
# tokens ("gameplay", "movie", "podcast") are kept forever as aliases
# so old jobs and old UI builds still route correctly.
_UI_TO_ENUM: dict[str, dict] = {
    # ── Legacy aliases (must work forever) ──
    "gameplay": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
    },
    "movie": {"content_type": ContentType.NARRATIVE},
    "podcast": {"content_type": ContentType.PODCAST},

    # ── Phase 2 forward-compat entries ──
    # The actual dropdown options land in Phase 2, but the normalizer
    # contract is stable from Phase 1 onward.
    "debate": {
        "content_type": ContentType.PODCAST,
        "panel": True,
    },
    "panel": {
        "content_type": ContentType.PODCAST,
        "panel": True,
    },
    "interview": {"content_type": ContentType.PODCAST},
    "vlog": {"content_type": ContentType.VLOG},
    "narrative": {"content_type": ContentType.NARRATIVE},
    "cinematic": {"content_type": ContentType.NARRATIVE},
    "anime": {
        "content_type": ContentType.ANIME,
        "animated": True,
    },
    "cartoon": {
        "content_type": ContentType.ANIME,
        "animated": True,
    },
    "music_video": {"content_type": ContentType.MUSIC_VIDEO},
    "gameplay_fps": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
    },
    "gameplay_moba": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
    },
    "gameplay_tps": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
    },
    "gameplay_racing": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
    },
    # Stream is GAMING + facecam. It must NOT take the face-skipping
    # fast path — Phase 7 handles the STACKED_GAMEPLAY layout routing
    # and the facecam portion still needs the regular face pipeline.
    "stream": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": False,
    },
    "sports": {"content_type": ContentType.SPORTS},

    # ── Special no-op tokens ──
    "": None,
    "auto": None,
    "unknown": None,
}


def normalize_ui_content_type(token: Optional[str]) -> Optional[NormalizedContentType]:
    """Normalize a UI content-type token to an enum + flag bundle.

    Accepts ``None``, the empty string, ``"auto"``, ``"unknown"``, and
    every registered UI dropdown token. Unknown / invalid tokens return
    ``None`` so the caller falls through to heuristic classification
    rather than crashing.

    Case-insensitive; leading / trailing whitespace is stripped. An
    unknown token emits a debug log so misconfigured dropdown values
    are traceable.
    """
    if token is None:
        return None
    key = token.strip().lower()
    if not key:
        return None
    if key in {"auto", "unknown"}:
        return None
    spec = _UI_TO_ENUM.get(key)
    if spec is None:
        logger.debug(
            "content_type_strings: unknown UI token %r — falling through "
            "to heuristic classification",
            token,
        )
        return None
    return NormalizedContentType(
        content_type=spec["content_type"],
        is_multi_speaker_panel=bool(spec.get("panel", False)),
        is_animated=bool(spec.get("animated", False)),
        anime_subtype=spec.get("anime_subtype"),
        music_subtype=spec.get("music_subtype"),
        is_gameplay_fastpath=bool(spec.get("gameplay_fastpath", False)),
        raw=key,
    )


def is_gameplay_override(token: Optional[str]) -> bool:
    """Return True when the UI token picks a pure gameplay variant.

    Used by the pipeline's early gameplay fast-path which skips face
    tracking entirely. Covers:

    - the legacy ``gameplay`` value
    - the Phase 2 ``gameplay_fps`` / ``gameplay_moba`` / ``gameplay_tps``
      / ``gameplay_racing`` variants

    Does **not** include ``stream`` — a stream is gaming-layout but has
    a facecam and still needs the face pipeline. Phase 7 routes stream
    content through ``ClipContentType.STREAM`` downstream.
    """
    n = normalize_ui_content_type(token)
    return n is not None and n.is_gameplay_fastpath


def is_user_override(token: Optional[str]) -> bool:
    """Return True when the UI token is a recognized, non-auto override.

    Used by the pipeline to decide whether to skip gameplay auto-detect
    (if the user already declared a non-gameplay type, we should not
    override their choice with a heuristic).
    """
    return normalize_ui_content_type(token) is not None
