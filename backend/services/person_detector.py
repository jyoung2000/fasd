"""Person body detector for the v4 AttentionAnchor fallback chain.

Why a separate module: 63% of faceless frames in dialogue / action footage
left the camera with nothing to anchor on after v3. A character with their
back turned, in profile, or far from camera produces no face but is clearly
locatable as a person bbox. Feeding those bboxes into the AttentionAnchor
priority chain (between "any face" and "last-face-decay") gives the
camera solver a real subject to lock onto instead of dropping into
last-face-decay or saliency-peak.

This module is a thin wrapper around the existing object_detector.py
YOLOv8n / MobileNet-SSD pipeline:
  - Reuses the singleton detector so we don't reload weights.
  - Filters to class_name == "person" + min confidence.
  - Computes `has_face` per detection by intersecting against the
    same-frame face list (the AttentionAnchor consumer de-prioritizes
    person bboxes whose face is already covered by the face anchor).

Returns a flat list[PersonRegion] (one per detection, multiple per frame
allowed) keyed by timestamp so callers can group by frame.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Confidence floor — YOLO sometimes emits low-conf person boxes for
# silhouettes / TV screens. 0.40 keeps recall on real subjects without
# letting noise into the anchor chain.
DEFAULT_PERSON_CONF = 0.40


@dataclass
class PersonRegion:
    """Single person body detection.

    Coordinates are normalized (0-1) with origin at top-left of the source
    frame. Width / height are absolute fractions of the source dimensions.
    """
    timestamp: float
    cx: float          # 0-1 normalized center x
    cy: float          # 0-1 normalized center y
    width: float       # 0-1 normalized width
    height: float      # 0-1 normalized height
    confidence: float  # 0-1 detection confidence
    has_face: bool     # True iff a same-frame face center lies inside this bbox


def _face_inside(face, person: PersonRegion) -> bool:
    """Return True if the face's nose lies within the person bbox.

    Face coordinates are 0-100 percent (face_detector convention); the
    PersonRegion is 0-1. Convert face to 0-1 and check containment.
    """
    fx = float(getattr(face, "nose_x", 50.0)) / 100.0
    fy = float(getattr(face, "nose_y", 50.0)) / 100.0
    half_w = person.width / 2.0
    half_h = person.height / 2.0
    return (
        person.cx - half_w <= fx <= person.cx + half_w
        and person.cy - half_h <= fy <= person.cy + half_h
    )


def detect_persons_in_frames(
    frame_paths: list,
    face_results: Optional[list] = None,
    conf_threshold: float = DEFAULT_PERSON_CONF,
    job_id: str = "",
) -> list:
    """Run YOLOv8n / MobileNet-SSD over frame_paths and return person bodies.

    Args:
        frame_paths: list of (timestamp, path) tuples — same shape the
            existing object_detector accepts. Pipeline.py constructs this
            from the frame_extractor output.
        face_results: optional list[FrameFaces] aligned with frame_paths
            (same length, same order) so we can mark has_face per detection.
        conf_threshold: minimum YOLO/SSD confidence to keep a detection.
        job_id: for logging.

    Returns:
        list[PersonRegion] — flat list, may contain multiple regions per
        frame. Empty if no detector backend is available.
    """
    # Lazy import so test files that don't need cv2 / ultralytics still
    # import this module without pulling in the heavy detector path.
    from backend.services.object_detector import get_detector
    import cv2  # local import — only used inside the detect loop

    detector = get_detector()
    if detector.backend_name == "none":
        logger.info(
            "[%s] PersonDetector: no backend available (object_detector=none)",
            job_id,
        )
        return []

    # Build a face index aligned with frame_paths so per-frame has_face
    # checks are O(faces in this frame), not O(all faces).
    face_by_index = {}
    if face_results:
        for i, ff in enumerate(face_results):
            face_by_index[i] = list(getattr(ff, "faces", []) or [])

    persons: list = []
    n_frames = 0
    n_with_face = 0
    n_without_face = 0
    for i, item in enumerate(frame_paths):
        # Accept (timestamp, path) tuples — same shape as object_detector
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            timestamp, path = item[0], item[1]
        else:
            continue
        n_frames += 1
        img = cv2.imread(str(path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        try:
            raw_detections = detector.detect(img, float(timestamp))
        except Exception as exc:
            logger.warning(
                "[%s] PersonDetector: detect failed at t=%.2f: %s",
                job_id, float(timestamp), exc,
            )
            continue

        # Filter to person class above conf threshold.
        for det in raw_detections:
            if getattr(det, "class_name", "") != "person":
                continue
            if float(getattr(det, "confidence", 0.0)) < conf_threshold:
                continue
            # ObjectDetection is in 0-100 percent, top-left origin via
            # bbox_center: cx, cy are centers, w/h are full widths in %.
            cx = float(det.x) / 100.0
            cy = float(det.y) / 100.0
            pw = float(det.w) / 100.0
            ph = float(det.h) / 100.0
            person = PersonRegion(
                timestamp=float(timestamp),
                cx=cx, cy=cy, width=pw, height=ph,
                confidence=float(det.confidence),
                has_face=False,
            )
            # has_face: any face in this frame whose nose lies inside the
            # person bbox. We use the index because face_results is aligned
            # with frame_paths.
            faces_here = face_by_index.get(i, [])
            for face in faces_here:
                if _face_inside(face, person):
                    person.has_face = True
                    break
            persons.append(person)
            if person.has_face:
                n_with_face += 1
            else:
                n_without_face += 1

    logger.info(
        "[%s] [PersonDetector] %d frames processed, %d person regions detected "
        "(%d with face, %d without)",
        job_id, n_frames, len(persons), n_with_face, n_without_face,
    )
    return persons
