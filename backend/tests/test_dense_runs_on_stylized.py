"""Phase 2 — dense face detection must run on stylized clips.

Two failures the pipeline used to have:

  - **Anime jobs**: sparse FaceMesh returns ~0 faces because the
    human-trained model rejects stylized characters. The dense
    pass was gated on ``face_results`` being non-empty, so the
    cascade augmentation never ran.
  - **Cartoon-shooter gameplay** (TF2, Overwatch, …): same
    failure mode, plus YuNet locks onto cartoon character heads
    and pollutes the registry with high-confidence false positives.

Phase 2 fixes both:

  1. The dense gate accepts ``face_results OR _early_stylized_hint``
     so anime / gameplay jobs enter the dense pass even with empty
     sparse results.
  2. ``detect_faces_dense`` now takes ``is_stylized=True`` and
     calls ``_demote_stylized_faces`` afterwards, multiplying
     YuNet detections that don't cluster across nearby frames by
     0.3 so the face registry drops them.

This test asserts those two contracts via static AST inspection
(the actual pipeline runtime requires ffmpeg / cv2 / a real video,
which is overkill for verifying a one-line gate) plus a direct
test of the ``_demote_stylized_faces`` helper.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import pytest


def _read_pipeline_source() -> str:
    with open("backend/services/pipeline.py") as f:
        return f.read()


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_early_stylized_hint_assigned_before_dense_gate():
    """``_early_stylized_hint`` must be assigned BEFORE the dense
    block reads it — otherwise the gate references an undefined name
    on the first stylized job."""
    src = _read_pipeline_source()
    tree = ast.parse(src)
    fn = _find_function(tree, "_run_analysis_inner")
    assert fn is not None, "_run_analysis_inner not found in pipeline.py"

    first_assign_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_early_stylized_hint":
                    if first_assign_line is None or node.lineno < first_assign_line:
                        first_assign_line = node.lineno

    assert first_assign_line is not None, (
        "_early_stylized_hint is never assigned at _run_analysis_inner scope"
    )

    first_load_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "_early_stylized_hint":
            if isinstance(node.ctx, ast.Load):
                if first_load_line is None or node.lineno < first_load_line:
                    first_load_line = node.lineno

    assert first_load_line is not None, (
        "_early_stylized_hint is assigned but never read"
    )
    assert first_load_line >= first_assign_line, (
        f"_early_stylized_hint read at line {first_load_line} BEFORE "
        f"its assignment at line {first_assign_line}"
    )


def test_dense_gate_includes_stylized_hint():
    """Locate the dense-face block and verify the gate uses
    ``_early_stylized_hint`` so anime / gameplay jobs with empty
    sparse face_results still enter the dense pass."""
    src = _read_pipeline_source()
    marker = "# ── Dense face detection (1fps, CPU-only) ──"
    assert marker in src, "dense-face block marker missing from pipeline.py"
    tail = src.split(marker, 1)[1]
    head = "\n".join(tail.splitlines()[:30])
    assert "_early_stylized_hint" in head, (
        "Dense face gate must reference _early_stylized_hint — "
        "anime / gameplay jobs with empty sparse face_results would "
        "skip the dense pass otherwise. Loosen the gate to:\n"
        "    if settings.SUBJECT_TRACKING_ENABLED and "
        "(face_results or _early_stylized_hint):"
    )
    assert "face_results" in head, (
        "Dense face gate must still reference face_results for the "
        "live-action path"
    )


def test_is_stylized_propagates_into_detect_faces_dense_call():
    """The pipeline must pass ``is_stylized=_early_stylized_hint``
    into ``detect_faces_dense`` so the demotion pass runs on
    stylized content."""
    src = _read_pipeline_source()
    assert "is_stylized=_early_stylized_hint" in src, (
        "detect_faces_dense must be called with "
        "is_stylized=_early_stylized_hint"
    )


def test_anime_override_sets_stylized_hint():
    """The override branch for ``content_type_override='anime'``
    must flip ``_early_stylized_hint`` True so anime jobs hit the
    cascade augmentation path."""
    src = _read_pipeline_source()
    # The hint logic should reference all four anime tokens
    for token in ("anime", "cartoon", "animation", "animated"):
        assert f'"{token}"' in src, f"missing anime token {token!r}"


def test_gameplay_override_sets_stylized_hint():
    """Gameplay overrides (gameplay / gameplay_fps / fps / moba /
    tps / racing / hero_shooter) must all flip ``_early_stylized_hint``
    so cartoon-shooter jobs hit the demotion pass."""
    src = _read_pipeline_source()
    for token in ("gameplay", "fps", "moba", "tps", "racing"):
        assert f'"{token}"' in src, f"missing gameplay token {token!r}"


def test_game_type_sets_stylized_hint():
    """When the user picked a specific game (e.g. 'tf2', 'valorant')
    via the dropdown, that alone should flip the stylized hint."""
    src = _read_pipeline_source()
    assert 'getattr(job, "game_type"' in src, (
        "pipeline must read job.game_type to set the stylized hint"
    )


# ──────────────────── _demote_stylized_faces helper tests ────────────────────


@dataclass
class _StubFace:
    confidence: float = 0.0
    is_human: bool = True
    identity_embedding: object = None
    nose_x: float = 50.0


@dataclass
class _StubFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def test_demote_stylized_faces_drops_unclustered_yunet():
    """Phase 2: a single isolated YuNet detection (no embedding
    cluster match nearby) gets its confidence multiplied by 0.3."""
    pytest.importorskip("numpy")
    import numpy as np
    from backend.services.face_detector import _demote_stylized_faces

    rng = np.random.RandomState(0)
    face1 = _StubFace(
        confidence=0.85, is_human=True,
        identity_embedding=rng.randn(128).tolist(),
    )
    face2 = _StubFace(
        confidence=0.85, is_human=True,
        identity_embedding=rng.randn(128).tolist(),  # different "person"
    )

    results = [
        _StubFrameFaces(timestamp=0.0, faces=[face1]),
        _StubFrameFaces(timestamp=2.0, faces=[face2]),
    ]
    n = _demote_stylized_faces(results)
    assert n == 2
    assert abs(face1.confidence - 0.255) < 1e-3
    assert abs(face2.confidence - 0.255) < 1e-3


def test_demote_stylized_faces_keeps_clustered_real_human():
    """A face that DOES cluster with another face in a nearby
    frame (same human across frames) is NOT demoted — it's a real
    person the embedding network found."""
    pytest.importorskip("numpy")
    import numpy as np
    from backend.services.face_detector import _demote_stylized_faces

    rng = np.random.RandomState(42)
    base = rng.randn(128)
    base = base / float(np.linalg.norm(base))
    embed_a = (base + 0.05 * rng.randn(128)).tolist()
    embed_b = (base + 0.05 * rng.randn(128)).tolist()

    face_a = _StubFace(confidence=0.85, identity_embedding=embed_a)
    face_b = _StubFace(confidence=0.85, identity_embedding=embed_b)

    results = [
        _StubFrameFaces(timestamp=0.0, faces=[face_a]),
        _StubFrameFaces(timestamp=0.5, faces=[face_b]),
    ]
    n = _demote_stylized_faces(results)
    assert n == 0
    assert face_a.confidence == 0.85
    assert face_b.confidence == 0.85


def test_demote_stylized_faces_skips_non_human_cascade_results():
    """Anime cascade detections are marked is_human=False — the
    demotion pass MUST leave them alone (otherwise we'd undo the
    whole point of running the cascade)."""
    pytest.importorskip("numpy")
    from backend.services.face_detector import _demote_stylized_faces

    cascade_face = _StubFace(
        confidence=0.85, is_human=False, identity_embedding=None,
    )
    results = [_StubFrameFaces(timestamp=0.0, faces=[cascade_face])]
    n = _demote_stylized_faces(results)
    assert n == 0
    assert cascade_face.confidence == 0.85


def test_demote_stylized_faces_handles_no_embeddings():
    """A live-action face with no embedding should still be demoted
    (we can't cluster it, so we treat it as suspect)."""
    from backend.services.face_detector import _demote_stylized_faces

    face = _StubFace(confidence=0.85, identity_embedding=None)
    results = [_StubFrameFaces(timestamp=0.0, faces=[face])]
    _demote_stylized_faces(results)
    # No embeddings → conservative demotion
    assert face.confidence < 0.85


def test_demote_stylized_faces_empty_input_returns_zero():
    from backend.services.face_detector import _demote_stylized_faces

    assert _demote_stylized_faces([]) == 0
