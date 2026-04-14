"""Week 2 Part A — unit tests for the dense-face anime augmentation.

Exercises ``_augment_dense_with_anime`` in isolation by mocking
``detect_anime_faces`` and feeding canned ``FrameFaces`` / frame-path
lists. The helper is the exact code path ``detect_faces_dense`` runs
when ``is_animated=True``, so these tests pin its contract without
needing ffmpeg / cv2 / a real video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest


# Shared stubs. ``FrameFaces`` from the real module pulls in an
# ``identity_embedding`` field that requires numpy to round-trip. The
# test helpers only touch ``.faces`` (a mutable list), ``.timestamp``,
# and ``.frame_path``, so importing the real class is fine — we just
# don't populate the embedding.


@dataclass
class _StubFace:
    """Minimal FaceInfo-shaped stub for tests that don't need the full dataclass."""
    confidence: float = 0.0
    is_human: bool = True
    x_center: float = 50.0
    y_center: float = 50.0
    width: float = 10.0
    height: float = 12.0


@dataclass
class _StubFrameFaces:
    timestamp: float
    frame_path: str
    faces: list = field(default_factory=list)


@dataclass
class _StubAnimeDetection:
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    source: str = "lbpcascade_animeface"


@dataclass
class _StubAnimeResult:
    timestamp: float
    detections: list = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def has_faces(self) -> bool:
        return bool(self.detections)


def _make_results_and_paths() -> tuple[list, list]:
    """Build three FrameFaces: weak, strong, empty."""
    results = [
        _StubFrameFaces(
            timestamp=0.0,
            frame_path="/tmp/frame_000.jpg",
            faces=[_StubFace(confidence=0.30)],  # weak — augment
        ),
        _StubFrameFaces(
            timestamp=0.5,
            frame_path="/tmp/frame_001.jpg",
            faces=[_StubFace(confidence=0.80)],  # strong — skip
        ),
        _StubFrameFaces(
            timestamp=1.0,
            frame_path="/tmp/frame_002.jpg",
            faces=[],                              # empty — augment
        ),
    ]
    frame_paths = [
        (fr.timestamp, fr.frame_path) for fr in results
    ]
    return results, frame_paths


def _fake_detect(frame_path: str, *, timestamp: float = 0.0):
    """Fake detect_anime_faces — returns two detections for every frame."""
    return _StubAnimeResult(
        timestamp=timestamp,
        detections=[
            _StubAnimeDetection(
                x_center=30.0, y_center=40.0,
                width=12.0, height=15.0, confidence=0.85,
            ),
            _StubAnimeDetection(
                x_center=70.0, y_center=55.0,
                width=10.0, height=13.0, confidence=0.90,
            ),
        ],
    )


def _fake_to_face_info(det):
    """Fake to_face_info — returns an object that walks like FaceInfo."""
    return _StubFace(
        confidence=float(det.confidence),
        is_human=False,     # Critical: anime detections must not trip the human verifier
        x_center=float(det.x_center),
        y_center=float(det.y_center),
        width=float(det.width),
        height=float(det.height),
    )


# ─────────────────────────────────────────────────────────────────────


def test_augment_appends_to_weak_and_empty_frames():
    """Augmentation touches frames 0 (weak) and 2 (empty), skips frame 1 (strong)."""
    from backend.services import face_detector

    results, frame_paths = _make_results_and_paths()

    with patch.object(
        face_detector, "_augment_dense_with_anime",
        wraps=face_detector._augment_dense_with_anime,
    ), patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert n_frames == 2, "weak + empty frames should be augmented"
    assert n_faces == 4, "two detections per augmented frame"

    # Frame 0 originally had one weak face; now has 1 weak + 2 anime = 3
    assert len(results[0].faces) == 3
    # Frame 1 was strong — untouched
    assert len(results[1].faces) == 1
    assert results[1].faces[0].confidence == 0.80
    # Frame 2 was empty; now has 2 anime faces
    assert len(results[2].faces) == 2

    # Every anime-origin face must be marked non-human so the pose
    # verifier downstream skips it.
    for anime_face in results[0].faces[1:]:  # all but the original weak face
        assert anime_face.is_human is False
    for anime_face in results[2].faces:
        assert anime_face.is_human is False


def test_flag_off_short_circuits():
    """USE_ANIME_FACE_DETECTOR=False → helper returns (0,0) without calling detector."""
    from backend.services import face_detector

    results, frame_paths = _make_results_and_paths()

    # Capture calls to detect_anime_faces so we can verify it's never hit.
    call_count = {"n": 0}

    def _spy(*args, **kwargs):
        call_count["n"] += 1
        return _StubAnimeResult(timestamp=0.0)

    with patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        False,
    ), patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_spy,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert (n_frames, n_faces) == (0, 0)
    assert call_count["n"] == 0, "detector must not run when flag is OFF"
    # Results are untouched
    assert len(results[0].faces) == 1
    assert len(results[1].faces) == 1
    assert len(results[2].faces) == 0


def test_strong_conf_threshold_respected():
    """A face at exactly the threshold (0.55) counts as strong and skips augmentation."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(
            timestamp=0.0,
            frame_path="/tmp/f.jpg",
            faces=[_StubFace(confidence=0.55)],  # exactly at threshold
        ),
        _StubFrameFaces(
            timestamp=0.5,
            frame_path="/tmp/g.jpg",
            faces=[_StubFace(confidence=0.54)],  # just below
        ),
    ]
    frame_paths = [(r.timestamp, r.frame_path) for r in results]

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, _ = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert n_frames == 1, "only the sub-threshold frame gets augmented"
    # The 0.55 frame stayed at 1 face; the 0.54 frame got +2 anime.
    assert len(results[0].faces) == 1
    assert len(results[1].faces) == 3


def test_mismatched_frame_paths_are_safe():
    """Helper doesn't crash when frame_paths is shorter than results."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(timestamp=0.0, frame_path="/tmp/a.jpg", faces=[]),
        _StubFrameFaces(timestamp=0.5, frame_path="/tmp/b.jpg", faces=[]),
    ]
    frame_paths = [(0.0, "/tmp/a.jpg")]  # only one entry — second should be skipped

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert n_frames == 1, "only index 0 was augmentable; index 1 out of range"
    assert n_faces == 2


def test_no_detections_returned_leaves_results_unchanged():
    """When detect_anime_faces returns no detections, the frame isn't touched."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(timestamp=0.0, frame_path="/tmp/a.jpg", faces=[]),
    ]
    frame_paths = [(0.0, "/tmp/a.jpg")]

    def _empty(frame_path, *, timestamp=0.0):
        return _StubAnimeResult(timestamp=timestamp, detections=[])

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_empty,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert (n_frames, n_faces) == (0, 0)
    assert len(results[0].faces) == 0


def test_to_face_info_exception_is_swallowed():
    """A single conversion failure shouldn't kill the whole augmentation pass."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(timestamp=0.0, frame_path="/tmp/a.jpg", faces=[]),
    ]
    frame_paths = [(0.0, "/tmp/a.jpg")]

    calls = {"n": 0}

    def _flaky(det):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated converter failure")
        return _fake_to_face_info(det)

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_flaky,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    # First detection was dropped but the second succeeded.
    assert n_frames == 1
    assert n_faces == 2  # count reflects attempted faces, not survivors
    assert len(results[0].faces) == 1  # only the successful conversion


def test_log_line_fires_with_counts(caplog):
    """The summary log line must include both counts."""
    import logging
    from backend.services import face_detector

    results, frame_paths = _make_results_and_paths()

    with caplog.at_level(logging.INFO, logger="backend.services.face_detector"), \
         patch(
             "backend.services.anime_face_detector.detect_anime_faces",
             side_effect=_fake_detect,
         ), patch(
             "backend.services.anime_face_detector.to_face_info",
             side_effect=_fake_to_face_info,
         ), patch(
             "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
             True,
         ):
        face_detector._augment_dense_with_anime(
            results, frame_paths, job_log_prefix="[job-xyz] ",
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Anime augmentation:" in m
        and "+4 faces" in m
        and "across 2 frames" in m
        and "[job-xyz]" in m
        for m in messages
    ), f"missing expected log line, got: {messages}"
