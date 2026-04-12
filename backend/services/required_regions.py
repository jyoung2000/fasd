"""Required-region adapter: reshapes ClipAI's existing face detection output
into the solver's input format.

Supports face-only mode (standalone solver) and full fusion mode (face +
object + saliency). The solver treats "required" regions as hard constraints
and "preferred" regions as tiebreakers.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequiredRegion:
    """A region that must stay inside the crop at a given timestamp."""
    timestamp: float
    cx: float            # normalized 0-1, center of the region
    cy: float            # normalized 0-1
    half_width: float    # normalized half-width
    half_height: float   # normalized half-height
    score: float         # 1.0 for active speaker, 0.55 passive (same shot), 0.8 passive (no speaker)
    tier: str = "required"   # "required" | "preferred"
    source: str = "face"     # "face" | "object" | "saliency"
    face_slot: int = -1
    saliency_score: float = 0.0  # original saliency score (0-1), preserved from tracker
    weight: float = 1.0  # camera-path data-term weight (see build_required_regions)
    is_active_speaker: bool = False  # True when this face is the active speaker


def _iou_normalized(a, b) -> float:
    """IoU between two regions (using cx/cy/half_width/half_height)."""
    a_l, a_r = a.cx - a.half_width, a.cx + a.half_width
    a_t, a_b = a.cy - a.half_height, a.cy + a.half_height
    b_l, b_r = b.cx - b.half_width, b.cx + b.half_width
    b_t, b_b = b.cy - b.half_height, b.cy + b.half_height

    ix_l, ix_r = max(a_l, b_l), min(a_r, b_r)
    iy_t, iy_b = max(a_t, b_t), min(a_b, b_b)
    if ix_r <= ix_l or iy_b <= iy_t:
        return 0.0
    inter = (ix_r - ix_l) * (iy_b - iy_t)
    area_a = (a_r - a_l) * (a_b - a_t)
    area_b = (b_r - b_l) * (b_b - b_t)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlaps_any(region, existing_regions, iou_threshold=0.3) -> bool:
    """Check if region overlaps any existing region above threshold."""
    for er in existing_regions:
        if _iou_normalized(region, er) > iou_threshold:
            return True
    return False


def _merge_overlapping_features(frame_regions: list) -> list:
    """Merge saliency features that overlap face/object features.

    For each saliency-source region, compute IoU against all face and
    object regions in the same frame. If IoU > 0.3 against any, absorb
    the saliency region into that other feature by adding its score
    (capped at 1.0) and dropping the standalone saliency entry.

    This prevents a saliency bbox overlapping a face bbox from over-
    constraining the crop solver.
    """
    if not frame_regions:
        return frame_regions

    non_sal = [r for r in frame_regions if r.source != "saliency"]
    sal = [r for r in frame_regions if r.source == "saliency"]

    if not sal or not non_sal:
        return frame_regions

    kept_sal = []
    for sr in sal:
        merged = False
        for nr in non_sal:
            iou = _iou_normalized(sr, nr)
            if iou > 0.3:
                nr.score = min(1.0, nr.score + sr.score)
                merged = True
                break
        if not merged:
            kept_sal.append(sr)

    return non_sal + kept_sal


def _is_dialogue_mode(content_type) -> bool:
    """True for content types where faces are load-bearing and background
    saliency should not compete: TALKING_HEAD, CINEMATIC_DIALOGUE, and
    ANIMATION_DIALOGUE (talking anime characters).
    """
    if content_type is None:
        return False
    # Match by string value so this works whether content_type is the enum
    # or its .value — callers pass both forms.
    val = getattr(content_type, "value", content_type)
    return val in ("talking_head", "cinematic_dialogue", "animation_dialogue")


def _is_animated_mode(content_type) -> bool:
    """True for content types that are animated (anime / cartoons).
    Used to tune the lip-aperture promotion threshold since anime neutral
    expressions sit around 0.05 aperture and need a higher bar.
    """
    if content_type is None:
        return False
    val = getattr(content_type, "value", content_type)
    return val in ("animation", "animation_dialogue")


@dataclass
class _FrameSaliencyAdapter:
    """FrameSaliency-compatible wrapper built from a group of
    SaliencyRegion objects sharing a timestamp.

    `blobs` is a list of (x, y, w, h) tuples in normalized 0-1
    coordinates with top-left origin — exactly the shape
    build_required_regions expects from saliency_detector's
    FrameSaliency output.
    """
    timestamp: float
    blobs: List[tuple]
    mean_score: float


def _saliency_regions_to_frame_saliency(saliency_regions: list) -> list:
    """Group a flat list of SaliencyRegion objects (from
    saliency_tracker.track_saliency_in_frames) into per-frame
    FrameSaliency-compatible adapters.

    SaliencyRegion coordinates are percentages (0-100) centered on the
    bbox. FrameSaliency consumers expect normalized 0-1 top-left
    (x, y, w, h). We convert and keep the top-3 regions per frame by
    saliency_score to cap per-frame saliency noise.

    Returns an empty list if the input isn't the expected shape —
    callers can then treat it as a no-op fallback.
    """
    if not saliency_regions:
        return []
    # Duck-type: SaliencyRegion has .x/.y/.w/.h/.saliency_score attrs.
    first = saliency_regions[0]
    if not all(hasattr(first, a) for a in ("x", "y", "w", "h", "saliency_score")):
        return []

    by_ts: dict = {}
    for sr in saliency_regions:
        key = round(float(sr.timestamp), 2)
        by_ts.setdefault(key, []).append(sr)

    adapters: list = []
    for ts, regs in by_ts.items():
        regs.sort(key=lambda r: float(r.saliency_score), reverse=True)
        top = regs[:3]
        blobs = []
        score_sum = 0.0
        for r in top:
            # Center-origin percent → top-left-origin normalized 0-1.
            w_norm = float(r.w) / 100.0
            h_norm = float(r.h) / 100.0
            x_norm = max(0.0, float(r.x) / 100.0 - w_norm / 2.0)
            y_norm = max(0.0, float(r.y) / 100.0 - h_norm / 2.0)
            blobs.append((x_norm, y_norm, w_norm, h_norm))
            score_sum += float(r.saliency_score)
        mean_score = score_sum / max(len(top), 1)
        adapters.append(_FrameSaliencyAdapter(
            timestamp=ts, blobs=blobs, mean_score=mean_score,
        ))
    return adapters


def _saliency_weight_for_content(content_type, sal_score: float):
    """Return (weight, min_area) tuple for saliency regions, per content type.

    Table (from the spec):
        TALKING_HEAD / CINEMATIC_DIALOGUE / ANIMATION_DIALOGUE:
            0.15 + 0.25·s, max 0.40, min_area 0.04
        ANIMATION / GAMEPLAY:   0.30 + 0.60·s (legacy), min_area 0.02
        STREAM / MUSIC_VIDEO / GENERIC: 0.25 + 0.45·s, max 0.70, min_area 0.03
    """
    val = getattr(content_type, "value", content_type) if content_type is not None else None
    if val in ("talking_head", "cinematic_dialogue", "animation_dialogue"):
        return (min(0.40, 0.15 + 0.25 * sal_score), 0.04)
    if val in ("animation", "gameplay"):
        return (0.30 + 0.60 * sal_score, 0.02)
    # STREAM / MUSIC_VIDEO / GENERIC / None
    return (min(0.70, 0.25 + 0.45 * sal_score), 0.03)


def build_required_regions(
    frame_faces: list,
    active_speaker_events: list = None,
    frame_objects: list = None,
    frame_saliency: list = None,
    content_type=None,
    shot_cuts: list = None,
    frame_persons: list = None,
) -> List[List[RequiredRegion]]:
    """Return per-frame lists of required regions.

    Face regions are always built. Object and saliency regions are added
    when their data is provided (populated conditionally by content type
    in the pipeline).

    Args:
        frame_faces: list[FrameFaces] from face_detector.py
        active_speaker_events: active speaker timeline
        frame_objects: list[list[ObjectDetection]] — one list per frame,
            or flat list indexed by timestamp
        frame_saliency: list[FrameSaliency] from saliency_detector OR a
            flat list[SaliencyRegion] from saliency_tracker (both shapes
            are accepted; SaliencyRegion lists are auto-converted to
            FrameSaliency-compatible adapters).
        content_type: ClipContentType for per-type weighting / saliency
            downrank / lead-room bias.
        shot_cuts: optional list of shot-cut timestamps. When provided
            the dense AttentionAnchor stream (dialogue modes) uses them
            to break the boxcar smoother at cut boundaries.

    Returns:
        list of lists — one inner list per frame.
    """
    def _active_event_at(timestamp: float):
        if not active_speaker_events:
            return None
        for ev in active_speaker_events:
            if ev.start <= timestamp <= ev.end and ev.slot_id >= 0:
                return ev
        return None

    # Index object detections by rounded timestamp
    obj_by_time = {}
    if frame_objects:
        for obj in frame_objects:
            key = round(getattr(obj, 'timestamp', 0), 2)
            obj_by_time.setdefault(key, []).append(obj)

    # Index saliency by rounded timestamp.
    #
    # `frame_saliency` may arrive as:
    #   (a) a flat list[SaliencyRegion] from saliency_tracker
    #       (.x/.y/.w/.h/.saliency_score in 0-100 percent), or
    #   (b) a list of FrameSaliency-like objects from saliency_detector
    #       (already have .blobs as 0-1 top-left tuples and .mean_score).
    #
    # When we receive (a) we convert to (b) so the existing .blobs
    # consumer below just works. Without this conversion the sal_by_time
    # index silently produces zero saliency regions.
    sal_by_time = {}
    _fs_used = frame_saliency
    if _fs_used:
        _first = _fs_used[0] if len(_fs_used) > 0 else None
        _has_blobs = _first is not None and hasattr(_first, "blobs")
        if not _has_blobs:
            _converted = _saliency_regions_to_frame_saliency(_fs_used)
            if _converted:
                _fs_used = _converted
                logger.info(
                    "[SaliencyFallback] generated %d frame_saliency entries "
                    "from saliency_regions",
                    len(_converted),
                )
    if _fs_used:
        for sf in _fs_used:
            sal_by_time[round(sf.timestamp, 2)] = sf

    dialogue_mode = _is_dialogue_mode(content_type)
    animated_mode = _is_animated_mode(content_type)

    # ── Attention anchor stream ──
    # For dialogue modes (live-action and animated), build a dense
    # per-frame AttentionAnchor timeline so frames with no face
    # RequiredRegion still have *something* for the camera solver to
    # anchor on. This is the Opus-level fallback: bridge faces across
    # 1.5s gaps, fall back to saliency peaks, and never leave a frame
    # without an anchor in the long run.
    anchor_by_time: dict = {}
    if dialogue_mode and frame_faces:
        try:
            from backend.services.attention_anchor import build_attention_anchors
            _anchors = build_attention_anchors(
                frame_faces=frame_faces,
                active_speaker_events=active_speaker_events,
                frame_saliency=_fs_used,
                shot_cuts=shot_cuts,
                frame_persons=frame_persons,
            )
            for a in _anchors:
                anchor_by_time[round(a.timestamp, 2)] = a
        except Exception as _ae:
            logger.info("[AnchorStream] build failed (non-fatal): %s", _ae)
            anchor_by_time = {}

    # ── Lip-aperture promotion threshold ──
    # Live-action neutral faces have lip_aperture ~0-0.03. Anime neutral
    # expressions can sit around 0.05 (drawn open-mouth style), so a
    # 0.05 threshold fires on listeners and wrongly promotes them. Bump
    # to 0.09 for animated content and require ≥3 consecutive frames to
    # hold before promoting (persist state across the per-frame loop).
    lip_thr = 0.09 if animated_mode else 0.05
    min_consec = 3 if animated_mode else 1
    # identity_id → consecutive count of frames with lip_aperture > lip_thr
    _lip_streak: dict = {}

    regions_per_frame = []
    _pre_merge_total = 0
    _post_merge_total = 0
    _active_count = 0
    _passive_same_count = 0
    _passive_no_speaker_count = 0
    _saliency_dropped_outside_face = 0
    _lead_room_applied = 0
    _hard_floor_boosts = 0

    for ff in frame_faces:
        active_ev = _active_event_at(ff.timestamp)
        active_slot = active_ev.slot_id if active_ev else -1
        on_screen = getattr(active_ev, "on_screen", True) if active_ev else True
        frame_regions = []
        ts_key = round(ff.timestamp, 2)

        # Is there any active speaker in this frame at all?
        frame_has_active_speaker = (active_slot >= 0 and on_screen)

        # For the tightened lip-aperture override: check whether any face in
        # this frame already has identity_id == active_slot.
        slot_ids_in_frame = {
            getattr(f, 'identity_id', -1) for f in ff.faces
            if getattr(f, 'is_human', True)
        }
        active_slot_visible_as_identity = (
            active_slot >= 0 and active_slot in slot_ids_in_frame
        )

        # ── Face regions (required tier) ──
        # Update lip streak counters for this frame (per identity) before
        # building regions, so the consecutive-frame rule can fire.
        _seen_this_frame = set()
        for face in ff.faces:
            if not getattr(face, 'is_human', True):
                continue
            fid = getattr(face, 'identity_id', -1)
            if fid < 0:
                continue
            _seen_this_frame.add(fid)
            if getattr(face, 'lip_aperture', 0) > lip_thr:
                _lip_streak[fid] = _lip_streak.get(fid, 0) + 1
            else:
                _lip_streak[fid] = 0
        # Reset streak for any identity that wasn't in this frame
        for fid in list(_lip_streak.keys()):
            if fid not in _seen_this_frame:
                _lip_streak[fid] = 0

        for face in ff.faces:
            if not getattr(face, 'is_human', True):
                continue
            slot_id = getattr(face, 'identity_id', -1)
            is_active = (slot_id >= 0 and slot_id == active_slot)
            # Lip-aperture override: only promote passive→active when no
            # other face in this frame already has identity_id == active_slot
            # (stops a laughing/yawning listener from shadow-promoting).
            # For animated content the threshold is 0.09 and the face must
            # have held open lips for ≥3 consecutive frames.
            if (not is_active
                    and slot_id >= 0
                    and getattr(face, 'lip_aperture', 0) > lip_thr
                    and _lip_streak.get(slot_id, 0) >= min_consec
                    and not active_slot_visible_as_identity):
                is_active = True

            # Off-screen speaker: treat every visible face as passive-same-shot.
            # The crop then anchors on the listener (reaction-shot framing)
            # instead of fabricating a speaker anchor.
            if active_ev is not None and not on_screen:
                is_active = False

            cx = face.nose_x / 100.0
            cy = face.nose_y / 100.0
            hw = (face.width / 100.0) / 2.0
            hh = (face.height / 100.0) / 2.0

            # Weight rules:
            #   active speaker:                          score=1.00 weight=1.40
            #   passive, same shot as active speaker:    score=0.55 weight=0.45
            #   passive, no active speaker in frame:     score=0.80 weight=0.80
            if is_active:
                score = 1.0
                weight = 1.4
                _active_count += 1
            elif frame_has_active_speaker or (active_ev is not None and not on_screen):
                # Another face in this frame is the speaker, OR the speaker
                # is off-screen (reaction shot) — demote heavily.
                score = 0.55
                weight = 0.45
                _passive_same_count += 1
            else:
                # No active speaker attributed to this frame — fall back to
                # the legacy "faces matter" behavior.
                score = 0.8
                weight = 0.8
                _passive_no_speaker_count += 1

            # ── Lead-room bias ──
            # Only the active speaker; requires a `yaw` attribute populated
            # upstream. Shifts cx by -0.05·sign(yaw) (in the direction the
            # subject is looking) and clamps to [hw, 1 - hw].
            if is_active:
                yaw = getattr(face, "yaw", None)
                if yaw is not None:
                    try:
                        yaw_val = float(yaw)
                        if abs(yaw_val) > 15.0:
                            sign = 1.0 if yaw_val > 0 else -1.0
                            cx = cx - 0.05 * sign
                            # Clamp so the region stays in-bounds.
                            lo = hw
                            hi = 1.0 - hw
                            if lo <= hi:
                                cx = max(lo, min(hi, cx))
                            _lead_room_applied += 1
                    except (TypeError, ValueError):
                        pass

            frame_regions.append(RequiredRegion(
                timestamp=ff.timestamp,
                cx=cx, cy=cy,
                half_width=hw, half_height=hh,
                score=score,
                tier="required",
                source="face",
                face_slot=slot_id,
                weight=weight,
                is_active_speaker=is_active,
            ))

        # ── Object regions: person class → required, non-face ──
        for obj in obj_by_time.get(ts_key, []):
            class_name = getattr(obj, 'class_name', '')
            if class_name != "person":
                continue
            cx = obj.x / 100.0
            cy = obj.y / 100.0
            hw = (obj.w / 100.0) / 2.0
            hh = (obj.h / 100.0) / 2.0
            candidate = RequiredRegion(
                timestamp=ff.timestamp,
                cx=cx, cy=cy,
                half_width=hw, half_height=hh,
                score=0.6,
                tier="required",
                source="object",
                weight=0.8,
            )
            if not _overlaps_any(candidate, frame_regions, 0.3):
                frame_regions.append(candidate)

        # ── Saliency regions: preferred tier ──
        sal_frame = sal_by_time.get(ts_key)
        if sal_frame:
            face_regions_this_frame = [
                r for r in frame_regions if r.source == "face"
            ]
            for blob in getattr(sal_frame, 'blobs', []):
                bx, by, bw, bh = blob[0], blob[1], blob[2], blob[3]
                area = bw * bh
                _sal_score = getattr(sal_frame, 'mean_score', 0.5)
                # Per-content-type weight + min-area gate
                sal_weight, min_area = _saliency_weight_for_content(
                    content_type, _sal_score,
                )
                if area < min_area:
                    continue
                cand_cx = bx + bw / 2
                cand_cy = by + bh / 2
                # Dialogue-mode gate: drop saliency regions whose center lies
                # outside the union of face bboxes expanded by 1.5× when
                # faces exist in the frame.
                if dialogue_mode and face_regions_this_frame:
                    inside_any = False
                    for fr_reg in face_regions_this_frame:
                        ehw = fr_reg.half_width * 1.5
                        ehh = fr_reg.half_height * 1.5
                        if (fr_reg.cx - ehw <= cand_cx <= fr_reg.cx + ehw
                                and fr_reg.cy - ehh <= cand_cy <= fr_reg.cy + ehh):
                            inside_any = True
                            break
                    if not inside_any:
                        _saliency_dropped_outside_face += 1
                        continue
                candidate = RequiredRegion(
                    timestamp=ff.timestamp,
                    cx=cand_cx, cy=cand_cy,
                    half_width=bw / 2, half_height=bh / 2,
                    score=_sal_score * 0.5,
                    tier="preferred",
                    source="saliency",
                    saliency_score=_sal_score,
                    weight=sal_weight,
                )
                if not _overlaps_any(candidate, frame_regions, 0.5):
                    frame_regions.append(candidate)

        # ── Hard floor: active speaker weight >= 1.5 × max other face weight ──
        # Guarantees the L1 camera solver's data term cannot pick a non-
        # speaking face over a speaking one, regardless of relative bbox
        # sizes. Only applies within the frame's face regions — saliency
        # and object regions are unaffected.
        face_regs = [r for r in frame_regions if r.source == "face"]
        active_regs = [r for r in face_regs if r.is_active_speaker]
        passive_regs = [r for r in face_regs if not r.is_active_speaker]
        if active_regs and passive_regs:
            max_other = max(r.weight for r in passive_regs)
            floor = 1.5 * max_other
            for ar in active_regs:
                if ar.weight < floor:
                    ar.weight = floor
                    _hard_floor_boosts += 1

        # ── Attention anchor fallback (dialogue modes) ──
        # If no face RequiredRegion survived for this frame, promote the
        # dense AttentionAnchor to a required region so the camera
        # solver always has something to lock onto. Without this the
        # crop drifts onto background motion during faceless action
        # beats (the primary symptom from the K S01E12 sanity run).
        if dialogue_mode and not any(r.source == "face" for r in frame_regions):
            anchor = anchor_by_time.get(ts_key)
            if anchor is not None:
                frame_regions.append(RequiredRegion(
                    timestamp=ff.timestamp,
                    cx=anchor.cx, cy=anchor.cy,
                    half_width=anchor.half_width, half_height=anchor.half_height,
                    score=max(0.3, min(0.9, 0.3 + 0.5 * anchor.confidence)),
                    tier="required",
                    source="saliency" if anchor.source in ("saliency_peak",) else "face",
                    weight=max(0.3, min(0.9, 0.3 + 0.5 * anchor.confidence)),
                    is_active_speaker=False,
                ))

        _pre_merge_total += len(frame_regions)
        frame_regions = _merge_overlapping_features(frame_regions)
        _post_merge_total += len(frame_regions)
        regions_per_frame.append(frame_regions)

    logger.info(
        "[SpeakerV2] weights: active=%d passive_same_shot=%d passive_no_speaker=%d",
        _active_count, _passive_same_count, _passive_no_speaker_count,
    )
    if _hard_floor_boosts > 0:
        logger.info(
            "[SpeakerV2] hard-floor boosted %d active-speaker weights "
            "(>= 1.5x passive in-frame max)",
            _hard_floor_boosts,
        )
    if _saliency_dropped_outside_face > 0:
        logger.info(
            "[SaliencyV2] dialogue-mode gate dropped %d saliency regions outside face bboxes",
            _saliency_dropped_outside_face,
        )
    if _lead_room_applied > 0:
        logger.info(
            "[SpeakerV2] lead-room bias applied to %d active-speaker regions",
            _lead_room_applied,
        )

    if _pre_merge_total != _post_merge_total:
        logger.info("[SaliencyParity] feature merge: %d → %d regions (-%d absorbed)",
                    _pre_merge_total, _post_merge_total,
                    _pre_merge_total - _post_merge_total)

    total_regions = sum(len(r) for r in regions_per_frame)
    frames_with = sum(1 for r in regions_per_frame if r)
    n_face = sum(1 for r in regions_per_frame for rr in r if rr.source == "face")
    n_obj = sum(1 for r in regions_per_frame for rr in r if rr.source == "object")
    n_sal = sum(1 for r in regions_per_frame for rr in r if rr.source == "saliency")
    logger.info("RequiredRegions: %d frames, %d with data, %d total "
                "(face=%d obj=%d sal=%d)",
                len(regions_per_frame), frames_with, total_regions,
                n_face, n_obj, n_sal)
    return regions_per_frame


def promote_preferred_to_required(
    regions_per_frame: List[List[RequiredRegion]],
    content_type=None,
) -> None:
    """For frames with no required regions, promote the highest-scoring
    preferred region to required.

    This is what makes anime and gameplay work — saliency becomes
    load-bearing exactly when faces are absent.

    No-op for TALKING_HEAD / CINEMATIC_DIALOGUE: for those modes a frame
    with no face means "wait" — let camera_path.py smooth through it
    rather than fabricating a new anchor from background saliency.
    """
    if _is_dialogue_mode(content_type):
        logger.info("[SaliencyV2] promote_preferred_to_required skipped (dialogue mode)")
        return
    promoted = 0
    for frame_regions in regions_per_frame:
        has_required = any(r.tier == "required" for r in frame_regions)
        if has_required or not frame_regions:
            continue
        preferred = [r for r in frame_regions if r.tier == "preferred"]
        if preferred:
            best = max(preferred, key=lambda r: r.score)
            best.tier = "required"
            promoted += 1
    if promoted > 0:
        logger.info("RequiredRegions: promoted %d preferred→required (no-face frames)",
                    promoted)
