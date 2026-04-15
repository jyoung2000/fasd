"""Lightweight face detection for subject tracking.

Uses OpenCV's DNN face detector (always available) with optional MediaPipe
upgrade. Runs on CPU — no GPU competition with Ollama/Whisper.
Processes 60 frames in ~1-2 seconds.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FaceInfo:
    """Face detection result for a single face."""
    x_center: float      # Horizontal center of face bbox, 0-100
    y_center: float      # Vertical center of face bbox, 0-100
    width: float          # Face bbox width as % of frame
    height: float         # Face bbox height as % of frame
    nose_x: float         # Best estimate of face center x, 0-100
    nose_y: float         # Best estimate of face center y, 0-100
    confidence: float     # Detection confidence, 0-1
    lip_aperture: float = 0.0  # Mouth openness ratio (0=closed, 1=wide open)
    identity_embedding: list = None  # 128-d face embedding for re-identification
    identity_id: int = -1       # Assigned face slot from registry (-1 = unassigned)
    is_speaking: bool = False   # Set by active speaker detection
    y_bottom: float = 0.0      # Bottom of face bbox as % of frame (for vertical positioning)
    is_human: bool = True       # Set by HumanFaceVerifier (default True = fail-open)
    pose_confidence: float = 0.0  # Pose verification confidence (0 if not verified)

    def __post_init__(self):
        """Ensure all numeric fields are native Python types, not numpy."""
        self.x_center = float(self.x_center)
        self.y_center = float(self.y_center)
        self.width = float(self.width)
        self.height = float(self.height)
        self.nose_x = float(self.nose_x)
        self.nose_y = float(self.nose_y)
        self.confidence = float(self.confidence)
        self.lip_aperture = float(self.lip_aperture)
        self.identity_id = int(self.identity_id)
        self.is_speaking = bool(self.is_speaking)
        self.y_bottom = float(self.y_bottom)


@dataclass
class FrameFaces:
    """All faces detected in a single frame."""
    timestamp: float
    frame_path: str
    faces: list[FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1  # Index of largest/most-prominent face


def _extract_face_embeddings(frame_img, faces_info: list, detector) -> list:
    """Extract 128-d identity embeddings using OpenCV SFace recognizer.

    Runs on CPU. ~5ms per face. No GPU competition with Whisper/Ollama.
    The embedding enables cross-frame re-identification: same person across
    different timestamps gets matched even if they move positions.
    """
    import cv2
    import numpy as np

    if not faces_info or detector is None:
        return faces_info

    recognizer = None
    sface_paths = [
        os.path.join(os.path.dirname(cv2.__file__), "data", "face_recognition_sface_2021dec.onnx"),
        "/usr/local/share/opencv4/face_recognition_sface_2021dec.onnx",
        os.path.join(os.path.dirname(__file__), "..", "models", "face_recognition_sface_2021dec.onnx"),
    ]
    sface_path = next((p for p in sface_paths if os.path.exists(p)), None)

    if sface_path is None:
        # Download one-time (~2MB)
        import urllib.request
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        sface_path = os.path.join(model_dir, "face_recognition_sface_2021dec.onnx")
        if not os.path.exists(sface_path):
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            logger.info("Downloading SFace recognition model...")
            try:
                urllib.request.urlretrieve(url, sface_path)
            except Exception as e:
                logger.warning("Failed to download SFace model: %s", e)
                return faces_info

    try:
        recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    except Exception as e:
        logger.warning("SFace recognizer unavailable: %s — skipping embeddings", e)
        return faces_info

    h, w = frame_img.shape[:2]
    for face in faces_info:
        try:
            # Convert percentage coords back to pixels for SFace
            fx = int((face.x_center - face.width / 2) * w / 100)
            fy = int((face.y_center - face.height / 2) * h / 100)
            fw = int(face.width * w / 100)
            fh = int(face.height * h / 100)

            # Build the aligned face for SFace (it expects the YuNet detection format)
            face_array = [fx, fy, fw, fh,
                          face.nose_x * w / 100, face.nose_y * h / 100,
                          0, 0, 0, 0, 0, 0, 0, 0, face.confidence]
            det = np.array([face_array], dtype=np.float32)

            aligned = recognizer.alignCrop(frame_img, det[0])
            embedding = recognizer.feature(aligned)
            face.identity_embedding = embedding.flatten().tolist()
            face.y_bottom = round((face.y_center + face.height / 2), 1)
        except Exception:
            pass  # Non-critical — face still usable without embedding

    return faces_info


def _detect_with_opencv_dnn(frame_paths, min_confidence, extract_embeddings=True):
    """Detect faces using OpenCV's built-in DNN face detector.

    Tries YuNet DNN detector first (much more accurate, gives nose/mouth landmarks),
    falls back to Haar cascade if YuNet model is unavailable.
    """
    import cv2

    results = []

    # Try YuNet DNN face detector first (much more accurate than Haar cascade)
    detector = None
    try:
        yunet_paths = [
            os.path.join(os.path.dirname(cv2.__file__), "data",
                         "face_detection_yunet_2023mar.onnx"),
            "/usr/local/share/opencv4/face_detection_yunet_2023mar.onnx",
            "/usr/share/opencv4/face_detection_yunet_2023mar.onnx",
            os.path.join(os.path.dirname(__file__), "..", "models",
                         "face_detection_yunet_2023mar.onnx"),
        ]
        yunet_path = next((p for p in yunet_paths if os.path.exists(p)), None)

        if yunet_path is None:
            # Download the model (one-time, ~350KB)
            import urllib.request
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)
            yunet_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
            if not os.path.exists(yunet_path):
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                logger.info("Downloading YuNet face detection model...")
                urllib.request.urlretrieve(url, yunet_path)
                logger.info("YuNet model downloaded to %s", yunet_path)

        if yunet_path and os.path.exists(yunet_path):
            detector = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                (320, 320),  # Will be resized per-frame
                min_confidence,
                0.3,  # NMS threshold
            )
            logger.info("Using OpenCV YuNet DNN face detector")
    except (cv2.error, AttributeError, Exception) as e:
        logger.debug("YuNet not available: %s", e)

    use_haar = detector is None

    if use_haar:
        # Haar cascade fallback — always available in OpenCV
        cascade_paths = [
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"),
        ]
        cascade_path = next((p for p in cascade_paths if os.path.exists(p)), None)
        if not cascade_path:
            logger.warning("No Haar cascade file found — face detection disabled")
            return None
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.warning("Failed to load Haar cascade — face detection disabled")
            return None
        logger.info("Using OpenCV Haar cascade face detector")

    for timestamp, path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
            continue

        h, w = img.shape[:2]
        faces: list[FaceInfo] = []

        if not use_haar and detector is not None:
            # YuNet DNN detection — much more accurate, gives landmarks
            detector.setInputSize((w, h))
            _, det_result = detector.detect(img)
            if det_result is not None:
                for face in det_result:
                    # YuNet returns: [x, y, w, h, right_eye_x, right_eye_y,
                    #   left_eye_x, left_eye_y, nose_x, nose_y,
                    #   right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y, score]
                    fx, fy, fw_px, fh_px = face[0], face[1], face[2], face[3]
                    nose_x_px, nose_y_px = face[8], face[9]
                    score = face[14]

                    cx = (fx + fw_px / 2) / w * 100
                    cy = (fy + fh_px / 2) / h * 100
                    fw_pct = fw_px / w * 100
                    fh_pct = fh_px / h * 100
                    nose_x_pct = nose_x_px / w * 100
                    nose_y_pct = nose_y_px / h * 100

                    # Compute lip aperture from mouth landmarks
                    right_mouth_y = face[11]
                    left_mouth_y = face[13]
                    mouth_center_y = (right_mouth_y + left_mouth_y) / 2
                    lip_aperture = abs(mouth_center_y - nose_y_px) / max(fh_px, 1) * 0.3

                    faces.append(FaceInfo(
                        x_center=float(round(cx, 1)),
                        y_center=float(round(cy, 1)),
                        width=float(round(fw_pct, 1)),
                        height=float(round(fh_pct, 1)),
                        nose_x=float(round(nose_x_pct, 1)),
                        nose_y=float(round(nose_y_pct, 1)),
                        confidence=float(round(float(score), 3)),
                        lip_aperture=float(round(lip_aperture, 3)),
                    ))

        elif use_haar:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30),
            )
            # NMS: remove overlapping detections (IoU > 0.3) to prevent
            # merged bounding boxes spanning multiple faces
            if len(detections) > 1:
                dets = sorted(detections.tolist() if hasattr(detections, 'tolist') else list(detections),
                              key=lambda f: f[2] * f[3], reverse=True)
                kept = [dets[0]]
                for det in dets[1:]:
                    x1, y1, w1, h1 = det
                    overlaps = False
                    for kx, ky, kw, kh in kept:
                        ix1 = max(x1, kx); iy1 = max(y1, ky)
                        ix2 = min(x1+w1, kx+kw); iy2 = min(y1+h1, ky+kh)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        union = w1*h1 + kw*kh - inter
                        if union > 0 and inter / union > 0.3:
                            overlaps = True
                            break
                    if not overlaps:
                        kept.append(det)
                detections = kept
            for (x, y, fw, fh) in detections:
                cx = (x + fw / 2) / w * 100
                cy = (y + fh / 2) / h * 100
                fw_pct = fw / w * 100
                fh_pct = fh / h * 100
                # Haar bboxes extend slightly more into background than toward
                # face center. Apply a small inward correction (max 3%) instead
                # of an aggressive heuristic that can overcorrect.
                edge_dist = abs(cx - 50)
                if edge_dist > 15:
                    pull = min(3.0, fw_pct * 0.15)
                    cx = cx + pull if cx < 50 else cx - pull
                faces.append(FaceInfo(
                    x_center=float(round(cx, 1)), y_center=float(round(cy, 1)),
                    width=float(round(fw_pct, 1)), height=float(round(fh_pct, 1)),
                    nose_x=float(round(cx, 1)), nose_y=float(round(cy, 1)),
                    confidence=0.8,  # Haar doesn't provide confidence
                ))

        # ── Reject merged detections ──
        # A single face wider than 18% of frame centered between 30-70%
        # is actually TWO faces merged into one box by Haar cascade.
        if len(faces) == 1 and faces[0].width > 18.0:
            if 30 < faces[0].x_center < 70:
                logger.debug(
                    "Frame %.1fs: rejecting merged face detection "
                    "(width=%.1f%%, center=%.1f%% — likely two speakers)",
                    timestamp, faces[0].width, faces[0].x_center,
                )
                faces = []

        # Extract identity embeddings (YuNet only — Haar doesn't provide landmarks)
        if extract_embeddings and not use_haar and detector is not None and faces and img is not None:
            faces = _extract_face_embeddings(img, faces, detector)

        primary = -1
        if faces:
            primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

        results.append(FrameFaces(
            timestamp=timestamp, frame_path=str(path),
            faces=faces, primary_face_idx=primary,
        ))

    # Explicitly release OpenCV resources
    if use_haar:
        del cascade
    elif detector is not None:
        del detector
    import gc
    gc.collect()

    return results


def _detect_with_facemesh(frame_paths, min_confidence, progress_callback=None):
    """Detect faces using MediaPipe FaceMesh — provides lip landmarks for active speaker detection.

    FaceMesh gives 468 landmarks per face including lip points.
    Lip Aperture Ratio = inner_lip_distance / face_height.
    When LAR > ~0.03, the person's mouth is open (likely speaking).
    Runs on CPU, ~3-5s for 60 frames.
    """
    import mediapipe as mp
    import cv2
    import time as _time

    face_mesh_module = None
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        face_mesh_module = mp.solutions.face_mesh
    if face_mesh_module is None:
        try:
            from mediapipe.python.solutions import face_mesh as fm_mod
            face_mesh_module = fm_mod
        except (ImportError, AttributeError):
            pass
    if face_mesh_module is None:
        logger.info("MediaPipe FaceMesh not available — falling back to FaceDetection")
        return None

    UPPER_LIP_INNER = 82
    LOWER_LIP_INNER = 87
    NOSE_TIP = 1

    results = []
    total_frames = len(frame_paths)
    _last_progress_emit = _time.monotonic()

    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,  # Support up to 10 speakers (was 4)
        refine_landmarks=True,
        min_detection_confidence=min_confidence,
    ) as mesh:
        for frame_idx, (timestamp, path) in enumerate(frame_paths):
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det_result = mesh.process(rgb)

            faces: list[FaceInfo] = []
            if det_result.multi_face_landmarks:
                for face_landmarks in det_result.multi_face_landmarks:
                    lm = face_landmarks.landmark

                    xs_px = [l.x * w for l in lm]
                    ys_px = [l.y * h for l in lm]
                    x_min, x_max = min(xs_px), max(xs_px)
                    y_min, y_max = min(ys_px), max(ys_px)
                    face_w = x_max - x_min
                    face_h = y_max - y_min

                    cx = ((x_min + x_max) / 2) / w * 100
                    cy = ((y_min + y_max) / 2) / h * 100
                    fw_pct = face_w / w * 100
                    fh_pct = face_h / h * 100

                    nose_x = lm[NOSE_TIP].x * 100
                    nose_y = lm[NOSE_TIP].y * 100

                    # Lip Aperture Ratio (LAR)
                    upper_lip_y = lm[UPPER_LIP_INNER].y * h
                    lower_lip_y = lm[LOWER_LIP_INNER].y * h
                    lip_distance = abs(lower_lip_y - upper_lip_y)
                    lip_aperture = lip_distance / max(face_h, 1)

                    faces.append(FaceInfo(
                        x_center=round(cx, 1),
                        y_center=round(cy, 1),
                        width=round(fw_pct, 1),
                        height=round(fh_pct, 1),
                        nose_x=round(nose_x, 1),
                        nose_y=round(nose_y, 1),
                        confidence=0.9,
                        lip_aperture=round(lip_aperture, 3),
                    ))

            # Reject merged detections
            if len(faces) == 1 and faces[0].width > 18.0:
                if 30 < faces[0].x_center < 70:
                    logger.debug("FaceMesh: rejecting merged detection w=%.1f%% c=%.1f%%",
                                 faces[0].width, faces[0].x_center)
                    faces = []

            primary = -1
            if faces:
                primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

            results.append(FrameFaces(
                timestamp=timestamp, frame_path=str(path),
                faces=faces, primary_face_idx=primary,
            ))

            # Emit per-frame progress every 50 frames or every 10 seconds
            if progress_callback and total_frames > 60:
                now = _time.monotonic()
                if (frame_idx + 1) % 50 == 0 or now - _last_progress_emit >= 10:
                    faces_so_far = sum(1 for r in results if r.faces)
                    progress_callback(
                        "facemesh_progress", frame_idx + 1, total_frames,
                        faces_so_far,
                    )
                    _last_progress_emit = now

    return results


def _detect_with_mediapipe(frame_paths, min_confidence):
    """Detect faces using MediaPipe — more accurate than Haar cascade.

    Handles both the legacy solutions API and newer API variants.
    """
    import mediapipe as mp
    import cv2

    # Try to find the right API
    face_detection_module = None
    FaceKeyPoint = None

    # Method 1: Legacy solutions API (mediapipe < 0.10.8)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
        face_detection_module = mp.solutions.face_detection
        FaceKeyPoint = face_detection_module.FaceKeyPoint

    # Method 2: Direct import (some versions)
    if face_detection_module is None:
        try:
            from mediapipe.python.solutions import face_detection as fd_mod
            face_detection_module = fd_mod
            FaceKeyPoint = fd_mod.FaceKeyPoint
        except (ImportError, AttributeError):
            pass

    if face_detection_module is None:
        logger.warning("MediaPipe face_detection API not available in this version")
        return None

    results = []
    get_key_point = getattr(face_detection_module, 'get_key_point', None)

    with face_detection_module.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence,
    ) as detector:
        for timestamp, path in frame_paths:
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det_result = detector.process(rgb)

            faces: list[FaceInfo] = []
            if det_result.detections:
                for det in det_result.detections:
                    bbox = det.location_data.relative_bounding_box
                    cx = (bbox.xmin + bbox.width / 2) * 100
                    cy = (bbox.ymin + bbox.height / 2) * 100
                    fw = bbox.width * 100
                    fh = bbox.height * 100

                    # Try to get nose tip for more accurate face center
                    nose_x, nose_y = cx, cy
                    if get_key_point and FaceKeyPoint:
                        try:
                            nose = get_key_point(det, FaceKeyPoint.NOSE_TIP)
                            if nose:
                                nose_x = nose.x * 100
                                nose_y = nose.y * 100
                        except Exception:
                            pass

                    conf = det.score[0] if det.score else 0.0
                    faces.append(FaceInfo(
                        x_center=round(cx, 1), y_center=round(cy, 1),
                        width=round(fw, 1), height=round(fh, 1),
                        nose_x=round(nose_x, 1), nose_y=round(nose_y, 1),
                        confidence=round(conf, 3),
                    ))

            primary = -1
            if faces:
                primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

            results.append(FrameFaces(
                timestamp=timestamp, frame_path=str(path),
                faces=faces, primary_face_idx=primary,
            ))

    return results


def _merge_detections(
    primary: list,  # list[FrameFaces] — FaceMesh results (have lip landmarks)
    secondary: list,  # list[FrameFaces] — YuNet results (may have more faces)
) -> list:
    """Merge face detections from two detectors.

    Keeps all primary (FaceMesh) faces — they have lip landmarks for active
    speaker detection. Adds secondary (YuNet) faces that don't overlap with
    any primary face (IoU by x_center distance > 10% of frame width).
    """
    merged = []
    for pri, sec in zip(primary, secondary):
        faces = list(pri.faces)  # Start with FaceMesh faces
        primary_xs = [f.x_center for f in faces]

        # Add non-overlapping secondary faces
        for sf in sec.faces:
            overlaps = any(abs(sf.x_center - px) < 10 for px in primary_xs)
            if not overlaps:
                faces.append(sf)
                primary_xs.append(sf.x_center)

        # Recompute primary face index
        primary_idx = -1
        if faces:
            primary_idx = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

        merged.append(FrameFaces(
            timestamp=pri.timestamp,
            frame_path=pri.frame_path,
            faces=faces,
            primary_face_idx=primary_idx,
        ))
    return merged


def _verify_faces_in_results(results: list, frame_paths: list) -> list:
    """Run human face verification on all detected faces.

    When USE_STRICT_POSE_MATCHING=true (default), uses per-frame strict
    containment: the pose nose must be INSIDE the face bbox.

    When USE_STRICT_POSE_MATCHING=false, uses the legacy per-face proximity
    check for backward compatibility.

    When USE_HUMAN_VERIFICATION=false or the verifier is unavailable,
    all faces are left as is_human=True (fail-open).
    """
    import os
    use_verification = os.environ.get(
        "USE_HUMAN_VERIFICATION", "true"
    ).lower() in ("true", "1", "yes")

    if not use_verification:
        logger.info("Human face verification disabled (USE_HUMAN_VERIFICATION=false)")
        return results

    use_strict = os.environ.get(
        "USE_STRICT_POSE_MATCHING", "true"
    ).lower() in ("true", "1", "yes")

    try:
        from backend.services.human_face_verifier import get_verifier
        verifier = get_verifier()
    except Exception as e:
        logger.warning("Human face verifier import failed: %s", e)
        return results

    if not verifier.available:
        logger.info("Human face verifier unavailable — all faces accepted (fail-open)")
        return results

    import cv2

    # Build a path lookup from frame_paths
    path_lookup = {round(t, 3): p for t, p in frame_paths}

    verified = 0
    non_human = 0
    for fr in results:
        if not fr.faces:
            continue

        # Try to load the frame image
        frame_path = path_lookup.get(round(fr.timestamp, 3), fr.frame_path)
        frame_bgr = cv2.imread(str(frame_path)) if frame_path else None
        if frame_bgr is None:
            continue

        if use_strict:
            # Strict mode: per-frame containment check
            h, w = frame_bgr.shape[:2]
            verifier.verify_faces_in_frame(frame_bgr, fr.faces, w, h)
            for face in fr.faces:
                verified += 1
                if not face.is_human:
                    non_human += 1
        else:
            # Legacy mode: per-face proximity check
            for face in fr.faces:
                is_human, pose_conf = verifier.verify_face(
                    frame_bgr,
                    face.nose_x,
                    face.nose_y,
                    face.width,
                    face.height,
                )
                face.is_human = is_human
                face.pose_confidence = pose_conf
                verified += 1
                if not is_human:
                    non_human += 1

    logger.info(
        "Human face verification (%s): %d faces checked, %d non-human detected",
        "strict" if use_strict else "legacy", verified, non_human,
    )
    return results


def detect_faces_batch(
    frame_paths: list[tuple[float, str]],
    min_confidence: float = 0.3,
    progress_callback=None,
) -> list[FrameFaces]:
    """Detect faces in extracted frames.

    Tries MediaPipe first (more accurate), falls back to OpenCV Haar cascade.
    When FaceMesh finds very few multi-face frames, supplements with YuNet
    to catch additional speakers that FaceMesh missed.
    Returns list of FrameFaces, one per input frame.
    """
    import time as _t
    t0 = _t.monotonic()

    final_results = None

    # Try FaceMesh first (gives lip landmarks for active speaker detection)
    facemesh_results = None
    try:
        facemesh_results = _detect_with_facemesh(frame_paths, min_confidence, progress_callback=progress_callback)
        if facemesh_results is not None:
            elapsed = _t.monotonic() - t0
            logger.info("Face detection using MediaPipe FaceMesh (%.1fs for %d frames)", elapsed, len(frame_paths))
            _log_summary(facemesh_results)

            # Always supplement FaceMesh with YuNet to catch faces that FaceMesh missed.
            with_faces = sum(1 for r in facemesh_results if r.faces)
            multi = sum(1 for r in facemesh_results if len(r.faces) >= 2)
            max_faces_per_frame = max((len(r.faces) for r in facemesh_results), default=0)

            if with_faces >= 5:
                try:
                    yunet_results = _detect_with_opencv_dnn(frame_paths, min_confidence)
                    if yunet_results is not None:
                        yunet_total = sum(len(r.faces) for r in yunet_results)
                        fm_total = sum(len(r.faces) for r in facemesh_results)
                        if yunet_total > fm_total:
                            merged = _merge_detections(facemesh_results, yunet_results)
                            new_total = sum(len(r.faces) for r in merged)
                            new_multi = sum(1 for r in merged if len(r.faces) >= 2)
                            logger.info(
                                "FaceMesh+YuNet merge: %d→%d total faces, multi-face frames %d→%d "
                                "(FaceMesh max %d/frame, YuNet found %d extra)",
                                fm_total, new_total, multi, new_multi,
                                max_faces_per_frame, new_total - fm_total,
                            )
                            _log_summary(merged)
                            final_results = merged
                        else:
                            logger.info(
                                "YuNet supplement: no extra faces (FaceMesh=%d, YuNet=%d)",
                                fm_total, yunet_total,
                            )
                except Exception as e:
                    logger.debug("YuNet supplement failed: %s", e)

            if final_results is None:
                final_results = facemesh_results
    except Exception as e:
        logger.info("FaceMesh unavailable (%s), trying FaceDetection", e)

    # Try MediaPipe FaceDetection (no lip landmarks but more robust)
    if final_results is None:
        try:
            results = _detect_with_mediapipe(frame_paths, min_confidence)
            if results is not None:
                elapsed = _t.monotonic() - t0
                logger.info("Face detection using MediaPipe FaceDetection (%.1fs for %d frames)", elapsed, len(frame_paths))
                _log_summary(results)
                final_results = results
        except Exception as e:
            logger.info("MediaPipe FaceDetection unavailable (%s), trying OpenCV", e)

    # Fall back to OpenCV (YuNet DNN → Haar cascade)
    if final_results is None:
        try:
            results = _detect_with_opencv_dnn(frame_paths, min_confidence)
            if results is not None:
                elapsed = _t.monotonic() - t0
                logger.info("Face detection using OpenCV DNN/Haar (%.1fs for %d frames)", elapsed, len(frame_paths))
                _log_summary(results)
                final_results = results
        except Exception as e:
            logger.warning("OpenCV face detection failed: %s", e)

    # All methods failed — return empty results (graceful degradation)
    if final_results is None:
        elapsed = _t.monotonic() - t0
        logger.warning("All face detection methods failed (%.1fs) — using AI estimates only", elapsed)
        final_results = [
            FrameFaces(timestamp=ts, frame_path=str(p))
            for ts, p in frame_paths
        ]

    # Run human face verification on all detected faces
    final_results = _verify_faces_in_results(final_results, frame_paths)

    return final_results


def _demote_stylized_faces(
    results: list, *, demotion_factor: float = 0.3, window: int = 5,
) -> int:
    """Demote YuNet face detections in stylized content.

    Cartoon-character faces (TF2, Overwatch, Marvel Rivals) and
    anime characters produce confident YuNet detections that pollute
    the registry: the camera then "tracks" a non-human face. When
    called on stylized content this helper multiplies each face's
    confidence by ``demotion_factor`` UNLESS the face has an SFace
    embedding cluster match across the next ``window`` frames (a
    real human face that re-appears) OR ``is_human`` is already
    False (an anime-cascade detection that the augmentation pass
    already marked).

    Returns the number of demoted faces. Operates in place on
    ``results`` so downstream code keeps the bbox coordinates but
    sees a low-confidence weight.
    """
    if not results:
        return 0

    try:
        import numpy as np
    except ImportError:
        return 0

    # Collect (frame_idx, face, embedding) for all live-action
    # faces with embeddings.
    indexed = []
    for fi, fr in enumerate(results):
        for face in getattr(fr, "faces", []) or []:
            if not getattr(face, "is_human", True):
                continue  # anime cascade detection — leave alone
            emb = getattr(face, "identity_embedding", None)
            if emb is None:
                indexed.append((fi, face, None))
                continue
            arr = np.asarray(emb, dtype=np.float32).flatten()
            n = float(np.linalg.norm(arr))
            if n <= 1e-9:
                indexed.append((fi, face, None))
                continue
            indexed.append((fi, face, arr / n))

    demoted = 0
    for i, (fi, face, ne) in enumerate(indexed):
        if ne is None:
            # No embedding — can't cluster; demote conservatively.
            try:
                face.confidence = float(face.confidence) * demotion_factor
                demoted += 1
            except Exception:
                continue
            continue
        matched = False
        for j, (fj, _other_face, oe) in enumerate(indexed):
            if i == j or oe is None:
                continue
            if abs(fj - fi) > window:
                continue
            sim = float(np.dot(ne, oe))
            if sim > 0.4:
                matched = True
                break
        if not matched:
            try:
                face.confidence = float(face.confidence) * demotion_factor
                demoted += 1
            except Exception:
                continue
    return demoted


def detect_faces_dense(
    video_path: str,
    start: float,
    end: float,
    sample_rate: float = 0.5,
    min_confidence: float = 0.5,
    extract_embeddings: bool = True,
    progress_callback=None,
    is_stylized: bool = False,
) -> list:
    """Dense face detection for a clip's time range.

    Unlike the pipeline's sparse detection (1 frame every ~5s), this extracts
    frames at 2 FPS and runs full face detection + embedding on each.
    Used at export time to build accurate per-frame face position data
    for layout decisions and smooth tracking.

    For a 30-second clip at 0.5s intervals = 60 frames.
    At ~10ms per frame (YuNet + SFace) = ~600ms total. Fast enough for export.

    Args:
        video_path: Source video file.
        start, end: Time range in seconds.
        sample_rate: Inter-frame interval in seconds.
        min_confidence: YuNet/FaceMesh confidence floor.
        extract_embeddings: Whether to merge YuNet+SFace embeddings.
        progress_callback: Optional callback for progress reporting.
        is_stylized: Phase 2 — when True, runs ``_demote_stylized_faces``
            after detection so YuNet hits on cartoon-character heads
            don't pollute the face registry. Caller is the pipeline,
            which passes the early stylized hint.
    """
    import re as _re
    import subprocess
    import tempfile
    import time as _time

    duration = end - start
    if duration <= 0:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y", "-threads", "2",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,
            "-vf", (
                f"fps=1/{sample_rate},"
                "scale='min(1920,iw)':'min(1080,ih)'"
                ":force_original_aspect_ratio=decrease"
            ),
            "-vsync", "vfr", "-q:v", "5",
            os.path.join(tmpdir, "dense_%04d.jpg"),
        ]
        # Scale timeout: ~0.5s per expected frame + 30s base for FFmpeg startup.
        # VP9 4K is ~3x slower to decode than H.264 1080p.
        expected_frames = int(duration / sample_rate)
        ffmpeg_timeout = max(120, int(30 + expected_frames * 0.5))
        logger.info(
            "[DenseFaces] Starting FFmpeg extraction: %.0fs video, %d expected frames, timeout=%ds",
            duration, expected_frames, ffmpeg_timeout,
        )
        try:
            # Use Popen to stream stderr and report frame extraction progress
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            _last_reported_frame = 0
            _last_progress_time = _time.monotonic()
            _frame_re = _re.compile(r'frame=\s*(\d+)')
            try:
                for line in proc.stderr:
                    match = _frame_re.search(line)
                    if match and progress_callback:
                        current_frame = int(match.group(1))
                        now = _time.monotonic()
                        if (current_frame - _last_reported_frame >= 50
                                or now - _last_progress_time >= 5):
                            pct = min(99, 100 * current_frame // max(expected_frames, 1))
                            progress_callback(
                                "extracting_frames", current_frame, expected_frames)
                            _last_reported_frame = current_frame
                            _last_progress_time = now
                proc.wait(timeout=ffmpeg_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                logger.warning(
                    "[DenseFaces] FFmpeg extraction timed out after %ds for %.0fs video (%d expected frames)",
                    ffmpeg_timeout, duration, expected_frames,
                )
                return []
            if proc.returncode != 0:
                logger.warning(
                    "[DenseFaces] FFmpeg extraction failed (rc=%d)",
                    proc.returncode,
                )
                return []
        except FileNotFoundError:
            logger.warning("[DenseFaces] FFmpeg not found")
            return []

        frame_files = sorted(
            f for f in os.listdir(tmpdir) if f.startswith("dense_")
        )
        if not frame_files:
            return []

        if progress_callback:
            progress_callback("extracting_done", len(frame_files), expected_frames)

        frame_paths = []
        for i, fname in enumerate(frame_files):
            ts = start + i * sample_rate
            frame_paths.append((ts, os.path.join(tmpdir, fname)))

        # Run face detection with FaceMesh
        if progress_callback:
            progress_callback("facemesh_start", 0, len(frame_paths))
        results = detect_faces_batch(
            frame_paths, min_confidence=min_confidence,
            progress_callback=progress_callback,
        )
        if progress_callback:
            progress_callback("facemesh_done", len(results), len(frame_paths))

        # If batch detection didn't produce embeddings (e.g. FaceMesh path),
        # supplement with YuNet+SFace for embeddings
        has_embeddings = any(
            f.identity_embedding is not None
            for fr in results for f in fr.faces
        )
        if not has_embeddings and extract_embeddings:
            if progress_callback:
                progress_callback("yunet_start", 0, len(frame_paths))
            yunet_results = _detect_with_opencv_dnn(
                frame_paths, min_confidence, extract_embeddings=True
            )
            if yunet_results:
                # Merge embeddings from YuNet into the main results
                for main_fr, yunet_fr in zip(results, yunet_results):
                    for mf in main_fr.faces:
                        if mf.identity_embedding is not None:
                            continue
                        # Find closest YuNet face by position
                        best_yf = None
                        best_dist = float('inf')
                        for yf in yunet_fr.faces:
                            dist = abs(mf.nose_x - yf.nose_x)
                            if dist < best_dist and yf.identity_embedding is not None:
                                best_dist = dist
                                best_yf = yf
                        if best_yf and best_dist < 10:
                            mf.identity_embedding = best_yf.identity_embedding

        # Phase 2: demote YuNet hits on cartoon / anime characters
        # when the caller signaled stylized content. Faces that
        # cluster across nearby frames via SFace embeddings are
        # left alone (real humans); singletons get their confidence
        # multiplied by 0.3 so the face registry treats them as
        # noise instead of building a slot around them.
        if is_stylized:
            n_demoted = _demote_stylized_faces(results)
            if n_demoted:
                logger.info(
                    "[DenseFaces] Stylized demotion: %d faces dropped to "
                    "0.3× confidence (no cluster match)",
                    n_demoted,
                )

        if progress_callback:
            progress_callback("complete", len(results), len(frame_paths))

        logger.info(
            "[DenseFaces] %d frames, %d with faces, %d with embeddings (%.1fs clip, %.1fs rate)",
            len(results),
            sum(1 for r in results if r.faces),
            sum(1 for r in results for f in r.faces if f.identity_embedding is not None),
            duration, sample_rate,
        )
        return results


def _log_summary(results: list[FrameFaces]):
    """Log face detection summary statistics."""
    total = sum(len(r.faces) for r in results)
    with_faces = sum(1 for r in results if r.faces)
    multi = sum(1 for r in results if len(r.faces) >= 2)
    logger.info(
        "Face detection: %d/%d frames have faces (%d total, %d multi-face)",
        with_faces, len(results), total, multi,
    )

    nose_xs = [
        r.faces[r.primary_face_idx].nose_x
        for r in results if r.faces and r.primary_face_idx >= 0
    ]
    if nose_xs:
        logger.info(
            "Face positions (nose_x): min=%.0f, max=%.0f, mean=%.1f, unique=%d",
            min(nose_xs), max(nose_xs),
            sum(nose_xs) / len(nose_xs),
            len(set(round(x) for x in nose_xs)),
        )

    # Lip aperture stats (for active speaker detection)
    all_lars = [f.lip_aperture for r in results for f in r.faces if f.lip_aperture > 0]
    if all_lars:
        logger.info(
            "Lip aperture: min=%.3f, max=%.3f, mean=%.3f, speaking_frames=%d (LAR>0.03)",
            min(all_lars), max(all_lars),
            sum(all_lars) / len(all_lars),
            sum(1 for l in all_lars if l > 0.03),
        )


# ── Gameplay Content Detection ───────────────────────────────────────────────


def detect_crosshair_persistence(sample_frame_paths: list[str]) -> float:
    """Look for a small high-contrast element at the exact center of frames.

    FPS games have a crosshair that stays fixed at the center while the
    background behind it changes.  We detect this by comparing temporal
    variance of the center 4×4 pixels vs the periphery of a 40×40 patch.

    Returns a score 0–1 indicating crosshair presence confidence.
    """
    import cv2
    import numpy as np

    if not sample_frame_paths:
        return 0.0

    center_patches = []
    for frame_path in sample_frame_paths[:30]:
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        cy, cx = h // 2, w // 2
        patch = img[cy - 20:cy + 20, cx - 20:cx + 20]
        if patch.shape == (40, 40):
            center_patches.append(patch)

    if len(center_patches) < 5:
        return 0.0

    # Center 4×4 should be stable (crosshair), periphery should vary (world moves)
    centers = np.stack([p[18:22, 18:22] for p in center_patches])
    center_temporal_var = np.var(centers, axis=0).mean()

    peripheries = np.stack([
        np.concatenate([p[:5, :].flatten(), p[-5:, :].flatten()])
        for p in center_patches
    ])
    periphery_temporal_var = np.var(peripheries, axis=0).mean()

    if periphery_temporal_var > 0:
        ratio = 1.0 - (center_temporal_var / periphery_temporal_var)
        return max(0.0, min(1.0, ratio))

    return 0.0


def detect_hud_corner_brightness(sample_frame_paths: list[str]) -> float:
    """Detect persistent high-saturation HUD elements in frame corners.

    Real-world footage has roughly uniform colour distribution.  Gameplay
    HUDs have brightly coloured, high-saturation overlays in fixed corner
    regions (top-right killfeed, bottom-center abilities, etc.).

    Returns a score 0–1 indicating HUD presence confidence.
    """
    import cv2
    import numpy as np

    if not sample_frame_paths:
        return 0.0

    corner_saturations: dict[str, list[float]] = {
        "tl": [], "tr": [], "bl": [], "br": [], "bc": [],
    }

    for frame_path in sample_frame_paths[:30]:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)

        ch, cw = h // 10, w // 10
        corner_saturations["tl"].append(float(sat[:ch, :cw].mean()))
        corner_saturations["tr"].append(float(sat[:ch, -cw:].mean()))
        corner_saturations["bl"].append(float(sat[-ch:, :cw].mean()))
        corner_saturations["br"].append(float(sat[-ch:, -cw:].mean()))
        corner_saturations["bc"].append(float(sat[-ch:, w // 2 - cw:w // 2 + cw].mean()))

    max_corner_score = 0.0
    for vals in corner_saturations.values():
        if vals:
            mean_sat = float(np.mean(vals))
            if mean_sat > 100:
                max_corner_score = max(max_corner_score, mean_sat / 255.0)

    return max_corner_score


def detect_stylization(sample_frame_paths: list[str]) -> float:
    """Return a 0-1 score for how 'cartoon/stylized' the frames look.

    Combines three cheap signals across up to 30 sample frames:

      - mean saturation > 130 (cartoon games + anime are high-sat,
        live-action footage rarely exceeds 100)
      - edge density > 0.10 (hard outlines + UI strokes vs. soft
        skin / hair gradients of real footage)
      - large uniform regions > 0.30 (flat color fills typical of
        cel-shaded characters and cartoon backgrounds)

    Returns 0.0 on missing dependencies (cv2 / numpy) so callers
    can fall back to the original face-only routing. Score is the
    average of the three booleans across the sampled frames, so:

      - 1.0 when all three signals fire on every sampled frame
        (TF2 / Overwatch / Marvel Rivals / anime)
      - 0.0 when none fire (live podcast, vlog, talking head)
      - ~0.4-0.6 for borderline content (saturated music videos,
        animated whiteboard explainers)

    The classifier uses 0.4 as the "is stylized" cutoff because
    real cartoon games consistently land above that and live
    action consistently lands below.
    """
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError:
        return 0.0

    if not sample_frame_paths:
        return 0.0

    high_sat_hits = 0
    high_edge_hits = 0
    flat_region_hits = 0
    n_scored = 0

    for frame_path in sample_frame_paths[:30]:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        n_scored += 1

        # Saturation signal — cartoon palettes are aggressively saturated
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        if mean_sat > 130:
            high_sat_hits += 1

        # Edge density signal — hard outlines / UI strokes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)
        if edge_density > 0.10:
            high_edge_hits += 1

        # Flat-region signal — large connected components of similar
        # color. We approximate via a coarse downsample + variance:
        # cel-shaded frames have low local variance over big tiles.
        small = cv2.resize(gray, (32, 18))
        # Compute the fraction of 4×4 tiles whose stdev is < 8 (flat).
        flat_tiles = 0
        total_tiles = 0
        for y in range(0, small.shape[0], 4):
            for x in range(0, small.shape[1], 4):
                tile = small[y:y + 4, x:x + 4]
                if tile.size < 4:
                    continue
                total_tiles += 1
                if float(np.std(tile)) < 8.0:
                    flat_tiles += 1
        if total_tiles > 0 and (flat_tiles / total_tiles) > 0.30:
            flat_region_hits += 1

    if n_scored == 0:
        return 0.0

    sat_score = high_sat_hits / n_scored
    edge_score = high_edge_hits / n_scored
    flat_score = flat_region_hits / n_scored
    return (sat_score + edge_score + flat_score) / 3.0


def build_classification_hint(
    decision: str,
    scores: dict,
    *,
    user_overridden: bool = False,
    anime_mode_detected: bool = False,
) -> dict:
    """Build the ``classification_hint`` dict for a job (Phase 3).

    Returns ``{}`` when no hint applies (clean classification, or
    user overridden — we don't second-guess explicit user choice).
    Otherwise returns a dict with:

      - ``suggested_content_type`` (``"gaming"`` or ``"anime"``)
      - ``suggested_game_type`` (``"generic_fps"`` or ``""``)
      - ``reason`` — one of ``"high_hud_score"``,
        ``"high_stylization"``, ``"anime_mode_detected"``,
        ``"high_face_rejection"``
      - ``scores`` — the raw signals so the UI can show a
        confidence indicator if it wants

    Decision rules:

      - If the user already picked a content type, return ``{}``
        — they win, no banner.
      - If the auto-router picked NOT gameplay AND the HUD score
        is > 0.4 OR stylization > 0.5: suggest ``"gaming"``.
      - If anime_mode_detected fired AFTER classification was
        locked to live-action: suggest ``"anime"``.
      - Otherwise: ``{}``.

    Advisory only — does NOT change analysis routing. The UI
    surfaces a banner with a CTA that triggers a re-analysis with
    the corrected ``content_type_override``.
    """
    if user_overridden:
        return {}

    hud = float(scores.get("hud", 0.0) or 0.0)
    face_ratio = float(scores.get("face_ratio", 0.0) or 0.0)
    stylization = float(scores.get("stylization", 0.0) or 0.0)
    crosshair = float(scores.get("crosshair", 0.0) or 0.0)

    # Anime suggestion — fires when the human-pose verifier
    # rejected most face detections AFTER classification ran.
    if anime_mode_detected and decision != "gameplay":
        return {
            "suggested_content_type": "anime",
            "suggested_game_type": "",
            "reason": "anime_mode_detected",
            "scores": {
                "hud": round(hud, 2),
                "face_ratio": round(face_ratio, 2),
                "stylization": round(stylization, 2),
                "crosshair": round(crosshair, 2),
            },
        }

    # Gaming suggestion — fires when auto-router didn't pick
    # gameplay but the HUD or stylization signals were close.
    if decision != "gameplay" and (hud > 0.4 or stylization > 0.5):
        return {
            "suggested_content_type": "gaming",
            "suggested_game_type": "generic_fps",
            "reason": "high_hud_score" if hud > 0.4 else "high_stylization",
            "scores": {
                "hud": round(hud, 2),
                "face_ratio": round(face_ratio, 2),
                "stylization": round(stylization, 2),
                "crosshair": round(crosshair, 2),
            },
        }

    return {}


def classify_gameplay_content_with_scores(
    dense_face_data: list,
    total_frames: int,
    sample_frame_paths: list[str],
) -> tuple[str, dict]:
    """Same as :func:`classify_gameplay_content` but returns ``(decision, scores)``.

    The ``scores`` dict contains the four signals the classifier
    computed (face_ratio, crosshair, hud, stylization) so the
    pipeline can populate ``job.classification_hint`` without
    re-running the helpers.
    """
    decision = classify_gameplay_content(
        dense_face_data, total_frames, sample_frame_paths,
    )
    scores = _last_classification_scores.copy()
    return decision, scores


# Module-level cache of the most recent classification's scores so
# ``classify_gameplay_content_with_scores`` can return them without
# re-running the four detectors. Populated by
# ``classify_gameplay_content`` on every call.
_last_classification_scores: dict = {}


def classify_gameplay_content(
    dense_face_data: list,
    total_frames: int,
    sample_frame_paths: list[str],
) -> str:
    """Classify whether video content is gameplay footage.

    Uses four signals:

      1. **face_ratio**       — fraction of dense frames with a
         live-action face. Live podcast clips run ~0.6-0.9; FPS
         gameplay runs ~0.0-0.1. Cartoon-shooter clips with huge
         character models can run 0.3-0.6 because YuNet locks
         onto the cartoon faces — that's why we need signal 4.
      2. **crosshair_score**  — temporal stability of the center
         4×4 patch. FPS clips with a fixed reticle score > 0.6.
      3. **hud_score**        — corner saturation persistence.
         Bright HUDs in any corner score > 0.5.
      4. **stylization_score** — high saturation + edge density +
         flat regions. Cartoon games and anime score > 0.4; live
         action scores < 0.2.

    Returns: ``'gameplay'`` | ``'unknown'`` | ``'not_gameplay'``.

    The rule:

      - Strong crosshair → always gameplay.
      - HUD + stylization → gameplay (TF2 / Overwatch / Apex
        cartoon-shooter fingerprint, regardless of face_ratio).
      - HUD + low face_ratio → gameplay (original live-action FPS
        gate, preserved verbatim for back-compat).
      - High face_ratio AND low stylization → not_gameplay (real
        live-action talking-head content).
      - Very low face_ratio → unknown (let other signals decide).
      - Otherwise → not_gameplay.
    """
    # Signal 1: face rarity
    frames_with_face = sum(
        1 for f in dense_face_data
        if hasattr(f, 'faces') and f.faces
    )
    face_ratio = frames_with_face / max(1, total_frames)

    # Signal 2: crosshair detection
    crosshair_score = detect_crosshair_persistence(sample_frame_paths)

    # Signal 3: HUD edge density
    hud_score = detect_hud_corner_brightness(sample_frame_paths)

    # Signal 4: cartoon / anime stylization
    stylization_score = detect_stylization(sample_frame_paths)

    logger.info(
        "Gameplay classification: face_ratio=%.2f, crosshair=%.2f, "
        "hud=%.2f, stylization=%.2f",
        face_ratio, crosshair_score, hud_score, stylization_score,
    )

    # Cache the four scores so callers can populate
    # ``job.classification_hint`` without re-running the detectors.
    _last_classification_scores.clear()
    _last_classification_scores.update({
        "face_ratio": round(face_ratio, 3),
        "crosshair": round(crosshair_score, 3),
        "hud": round(hud_score, 3),
        "stylization": round(stylization_score, 3),
    })

    # Strong gameplay signals win regardless of face_ratio.
    if crosshair_score > 0.6:
        return "gameplay"
    # Cartoon-shooter fingerprint: HUD + stylization. This is the
    # TF2 / Overwatch / Marvel Rivals route — characters look like
    # faces to YuNet so face_ratio is high, but the HUD + cartoon
    # palette confirm it's gameplay.
    if hud_score > 0.6 and stylization_score > 0.4:
        return "gameplay"
    # Original live-action FPS gate (low HUD threshold but requires
    # near-zero faces). Kept verbatim so existing test fixtures pass.
    if hud_score > 0.5 and face_ratio < 0.10:
        return "gameplay"

    # Real talking-head clips: lots of faces, no cartoon palette.
    if face_ratio > 0.30 and stylization_score < 0.4:
        return "not_gameplay"

    if face_ratio < 0.05:
        return "unknown"

    return "not_gameplay"
