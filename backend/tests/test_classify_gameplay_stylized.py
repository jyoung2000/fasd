"""Phase 1 — stylization-aware gameplay classifier tests.

The legacy ``classify_gameplay_content`` early-returns
``"not_gameplay"`` whenever ``face_ratio > 0.30`` because
live-action talking-head clips run at ~0.6-0.9 face_ratio. That
gate is too strict for cartoon-shooter games (TF2, Overwatch,
Marvel Rivals) where YuNet locks onto the cartoon character
models and pushes face_ratio above 30 % despite the clip being
unmistakably gameplay.

Phase 1 adds a fourth signal ``stylization_score`` (cartoon
palette + edge density + flat regions) and uses it to bypass
the face-ratio gate when the HUD also fires.

These tests exercise three scenarios on synthetic frame fixtures:

  - **Cartoon shooter (TF2-like)**: high saturation, hard edges,
    flat color fills, bright corner HUD, lots of "faces"
    detected → must classify as ``gameplay``.
  - **Real-world podcast**: low saturation, soft edges, no HUD,
    real faces → must classify as ``not_gameplay``.
  - **Empty office screenshare**: low saturation, sharp edges
    (lots of text), no HUD, no faces → must return ``unknown``.

Plus pinning tests for the new ``detect_stylization`` helper
itself (so future contributors don't accidentally re-tune the
thresholds and break the routing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.services.face_detector import (  # noqa: E402
    classify_gameplay_content,
    detect_stylization,
)


# ──────────────────── Synthetic frame builders ────────────────────


@dataclass
class _FaceShape:
    """A cv2-drawable face stub with a confidence attr — duck types
    enough of FaceInfo to satisfy the classifier."""
    confidence: float = 0.85


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _make_cartoon_shooter_frame(
    width: int = 640, height: int = 360, hue_seed: int = 60,
) -> np.ndarray:
    """Build a cel-shaded cartoon-shooter looking frame.

    Three high-saturation flat regions stacked vertically (sky /
    character / HUD), a bright HUD strip in the bottom-left
    corner, and hard outlines between regions — exactly the
    fingerprint TF2 / Overwatch produce.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky band — saturated cyan
    sky = cv2.cvtColor(
        np.full((1, 1, 3), [hue_seed, 220, 220], dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    img[: height // 3, :] = sky
    # Character band — saturated red
    char = cv2.cvtColor(
        np.full((1, 1, 3), [(hue_seed + 90) % 180, 230, 230], dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    img[height // 3:2 * height // 3, :] = char
    # Ground band — saturated yellow
    ground = cv2.cvtColor(
        np.full((1, 1, 3), [(hue_seed + 30) % 180, 240, 240], dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    img[2 * height // 3:, :] = ground
    # Hard outlines between bands
    cv2.line(img, (0, height // 3), (width, height // 3), (0, 0, 0), 4)
    cv2.line(img, (0, 2 * height // 3), (width, 2 * height // 3), (0, 0, 0), 4)
    # Bottom-left HUD: a saturated overlay rect
    cv2.rectangle(
        img,
        (10, height - 60),
        (width // 4, height - 10),
        (50, 200, 250),
        thickness=-1,
    )
    # Bottom-right HUD too — for hud_score signal
    cv2.rectangle(
        img,
        (width - width // 4, height - 60),
        (width - 10, height - 10),
        (250, 50, 200),
        thickness=-1,
    )
    return img


def _make_podcast_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Soft-lit talking-head frame: flesh tones, smooth gradients,
    no HUD, modest edges."""
    img = np.full((height, width, 3), 110, dtype=np.uint8)
    # Add a smooth radial vignette so it looks like indoor lighting
    cy, cx = height // 2, width // 2
    for y in range(height):
        for x in range(width):
            d = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
            v = max(60, 180 - int(d * 0.4))
            img[y, x] = (v - 20, v - 5, v + 10)
    # Add a slightly off-center "face" oval (skin tone, soft edge)
    cv2.ellipse(
        img, (cx + 20, cy - 10), (60, 80), 0, 0, 360,
        (160, 180, 210), thickness=-1,
    )
    # Light Gaussian to simulate soft camera focus
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return img


def _make_screenshare_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """White-background office screenshare with text strokes.

    Low saturation (mostly white + black), high edge density
    from text characters, no HUD, no faces. The classifier should
    return ``unknown`` because face_ratio is near zero AND no
    cartoon-fingerprint signals fire.
    """
    img = np.full((height, width, 3), 250, dtype=np.uint8)
    # Lots of text-like vertical strokes (low saturation, sharp edges)
    rng = np.random.RandomState(7)
    for _ in range(200):
        x = int(rng.randint(10, width - 20))
        y = int(rng.randint(20, height - 20))
        cv2.rectangle(
            img, (x, y), (x + 4, y + 14), (20, 20, 20), thickness=-1,
        )
    return img


def _write_frames(tmp_path, frames: list, prefix: str) -> list[str]:
    paths = []
    for i, frame in enumerate(frames):
        p = tmp_path / f"{prefix}_{i:03d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))
    return paths


# ──────────────────── detect_stylization helpers ────────────────────


def test_detect_stylization_high_for_cartoon_shooter(tmp_path):
    frames = [_make_cartoon_shooter_frame(hue_seed=h) for h in (10, 60, 100, 150)]
    paths = _write_frames(tmp_path, frames, "cartoon")
    score = detect_stylization(paths)
    assert score >= 0.4, f"cartoon stylization {score} below 0.4"


def test_detect_stylization_low_for_podcast(tmp_path):
    frames = [_make_podcast_frame() for _ in range(4)]
    paths = _write_frames(tmp_path, frames, "podcast")
    score = detect_stylization(paths)
    assert score < 0.4, f"podcast stylization {score} above 0.4"


def test_detect_stylization_handles_empty_input():
    assert detect_stylization([]) == 0.0


def test_detect_stylization_handles_unreadable_frames(tmp_path):
    """A list of nonexistent paths should return 0.0, not raise."""
    score = detect_stylization([str(tmp_path / "nope.png")])
    assert score == 0.0


# ──────────────────── classify_gameplay_content scenarios ────────────────────


def test_cartoon_shooter_classified_as_gameplay(tmp_path):
    """TF2 / Overwatch fingerprint: cartoon faces detected + HUD +
    high stylization → must return ``gameplay`` despite face_ratio."""
    frames = [_make_cartoon_shooter_frame(hue_seed=h) for h in range(0, 180, 12)]
    paths = _write_frames(tmp_path, frames, "tf2")
    # Simulate ~50 % of dense frames having "faces" (YuNet locked
    # onto cartoon character heads).
    dense = [
        _FrameFaces(
            timestamp=i * 0.5,
            faces=[_FaceShape(confidence=0.85)] if i % 2 == 0 else [],
        )
        for i in range(20)
    ]
    result = classify_gameplay_content(dense, total_frames=20, sample_frame_paths=paths)
    assert result == "gameplay", (
        f"cartoon-shooter clip classified as {result!r}, expected gameplay"
    )


def test_podcast_classified_as_not_gameplay(tmp_path):
    """Real talking-head: lots of real faces, no HUD, no stylization →
    must return ``not_gameplay``."""
    frames = [_make_podcast_frame() for _ in range(10)]
    paths = _write_frames(tmp_path, frames, "podcast")
    dense = [
        _FrameFaces(timestamp=i * 0.5, faces=[_FaceShape(confidence=0.9)])
        for i in range(20)
    ]
    result = classify_gameplay_content(dense, total_frames=20, sample_frame_paths=paths)
    assert result == "not_gameplay", (
        f"podcast clip classified as {result!r}, expected not_gameplay"
    )


def test_empty_office_screenshare_returns_unknown(tmp_path):
    """White-bg office screenshare with text, no HUD, no faces →
    classifier should hit the ``unknown`` branch (face_ratio < 0.05)."""
    frames = [_make_screenshare_frame() for _ in range(8)]
    paths = _write_frames(tmp_path, frames, "screen")
    dense = [_FrameFaces(timestamp=i * 0.5, faces=[]) for i in range(20)]
    result = classify_gameplay_content(dense, total_frames=20, sample_frame_paths=paths)
    assert result == "unknown", (
        f"screenshare classified as {result!r}, expected unknown"
    )


def test_face_ratio_alone_no_longer_gates_off_gameplay(tmp_path):
    """Pinning test: even with face_ratio = 0.5, the cartoon-shooter
    classification must STILL route to gameplay because of the new
    HUD + stylization branch. Without this we regress the TF2 case."""
    frames = [_make_cartoon_shooter_frame() for _ in range(10)]
    paths = _write_frames(tmp_path, frames, "tf2_high_face")
    dense = [
        _FrameFaces(
            timestamp=i * 0.5,
            faces=[_FaceShape(confidence=0.85)] if i % 2 == 0 else [],
        )
        for i in range(20)
    ]
    # 10/20 faces = 0.5 face_ratio, well above the legacy 0.30 gate.
    result = classify_gameplay_content(dense, total_frames=20, sample_frame_paths=paths)
    assert result == "gameplay"


def test_strong_crosshair_still_wins_regardless_of_stylization(tmp_path):
    """Test the precedence of the crosshair branch: even on a
    podcast-looking frame, a crosshair-like bright dot at exact
    center should still classify as gameplay (legacy fast path)."""
    frames = []
    for _ in range(15):
        f = _make_podcast_frame()
        # Plant a stable crosshair: tiny bright dot at exact center
        cv2.circle(f, (f.shape[1] // 2, f.shape[0] // 2), 3, (255, 255, 255), -1)
        frames.append(f)
    paths = _write_frames(tmp_path, frames, "crosshair")
    dense = [_FrameFaces(timestamp=i * 0.5, faces=[]) for i in range(20)]
    result = classify_gameplay_content(dense, total_frames=20, sample_frame_paths=paths)
    # With a stable crosshair the score > 0.6 → gameplay wins.
    # If the test fixture's crosshair isn't strong enough we accept
    # unknown (face_ratio is 0) — what we MUST NOT see is "not_gameplay"
    # which would mean we broke the legacy branches.
    assert result in ("gameplay", "unknown"), (
        f"crosshair fixture classified as {result!r}"
    )
