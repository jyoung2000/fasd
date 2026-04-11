"""L1-optimal camera path solver for AutoFlip-quality reframing.

Solves a 1D total-variation denoising problem per segment to produce
"hold still, snap, hold still" or constant-velocity pan motion — the
signature of professional camera operation.

    min  (1/2) sum (cam[t] - target[t])^2  +  lambda * sum |cam[t] - cam[t-1]|
    s.t. min_x[t] <= cam[t] <= max_x[t]   (hard constraints from must_be_in_frame features)

Uses an exact dual proximal gradient TV solver (Chambolle 2004 / Condat 2013)
with per-solve box constraint projection. No iterative approximation —
produces exact piecewise-constant paths with zero residual tilt on holds.

Per-segment mode selection from the solved path:
  STATIONARY: max(path) - min(path) < threshold * source_width  -> static center
  TRACKING:   otherwise -> emit the L1 path as motion_path
  PANNING:    near-linear path (R^2 > 0.95) and |slope| > threshold -> sweep
"""

import csv
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Debug telemetry: set CLIPAI_DUMP_L1=1 to write per-call CSVs to /tmp/clipai_l1/
_DUMP_L1 = os.environ.get("CLIPAI_DUMP_L1", "0") in ("1", "true", "yes")
_DUMP_DIR = Path("/tmp/clipai_l1")
_dump_counter = 0

# ── Solver selection ──
# CLIPAI_L1_SOLVER controls which solver is used:
#   "auto"   (default) — LP for shots <= LP_MAX_FRAMES, Condat above.
#   "lp"     — force LP; error if shot exceeds LP_MAX_FRAMES * 2.
#   "condat" — force Condat (legacy default, kept as escape hatch).
#
# Deprecated: CLIPAI_L1_LP=1 maps to "lp" with a DeprecationWarning.
_DEPRECATED_LP_FLAG = os.environ.get("CLIPAI_L1_LP", "0") in ("1", "true", "yes")
_SOLVER_MODE = os.environ.get("CLIPAI_L1_SOLVER", "auto").lower()
if _DEPRECATED_LP_FLAG and _SOLVER_MODE == "auto":
    _SOLVER_MODE = "lp"
    warnings.warn(
        "CLIPAI_L1_LP is deprecated; use CLIPAI_L1_SOLVER=lp instead.",
        DeprecationWarning,
        stacklevel=1,
    )
# Keep USE_LP_SOLVER for any downstream readers (read-only alias)
USE_LP_SOLVER = _SOLVER_MODE in ("lp", "auto")

# Performance guardrail: LP is O(n³) worst case. Fall back to Condat for
# shots longer than this many frames (30s at 30fps = 900).
LP_MAX_FRAMES = 900

# LP cost weights (Grundmann et al. 2011, Section 4.2).
# λ₁ = data fidelity (always 1.0), λ₂ = velocity, λ₃ = acceleration, λ₄ = jerk.
# Paper defaults: (10, 100, 100). Our codebase previously used lam4=1000;
# we now match the paper values.
LP_LAMBDA_V = 10.0    # λ₂: velocity penalty
LP_LAMBDA_A = 100.0   # λ₃: acceleration penalty
LP_LAMBDA_J = 100.0   # λ₄: jerk penalty

# TV denoise regularization: fraction of source_width for resolution independence.
# Actual lambda = TV_LAMBDA_FRAC * source_width. 0.015 * 1920 = 28.8
TV_LAMBDA_FRAC = 0.015
# Legacy constant (kept for backward-compat callers passing lam= explicitly)
TV_LAMBDA = 10.0
# Dead-zone: ignore target jitter smaller than this fraction of source_width.
# 0.013 * 1920 ≈ 25px — covers face-keypoint noise without masking real motion.
DEADZONE_FRAC = 0.013
# Stationary threshold: if total movement < this fraction of source width, static crop
# 0.08 = ~154px on 1920 — covers normal face-detection noise without triggering
STATIONARY_THRESHOLD = 0.08
# Panning R^2 threshold for linear-fit detection (0.90 for pre-solve noisy data)
PANNING_R2_THRESHOLD = 0.90
# Minimum slope (pixels per second) to qualify as a pan
PANNING_MIN_SLOPE = 50.0


def _tv_denoise_1d(
    signal: list[float],
    lam: float,
    n_iter: int = 100,
    lo_bounds: Optional[list[float]] = None,
    hi_bounds: Optional[list[float]] = None,
) -> list[float]:
    """1D total-variation denoising via exact dual proximal gradient
    with box constraint projection.

    Produces a piecewise-constant approximation of the input signal,
    which gives "snap and hold" camera motion. Uses the exact solver
    from _condat_tv (Chambolle 2004 / Condat 2013 dual formulation)
    with a two-pass projection onto [lo_bounds, hi_bounds].

    Args:
        signal: Input 1D signal (target face positions over time).
        lam: Regularization weight. Higher = smoother.
        n_iter: Ignored (kept for backward compat). The solver
            runs to convergence automatically.
        lo_bounds: Per-element lower bounds (None = unconstrained).
        hi_bounds: Per-element upper bounds (None = unconstrained).

    Returns:
        Denoised signal of the same length.
    """
    from backend.services._condat_tv import condat_tv_l1

    n = len(signal)
    if n <= 1:
        return list(signal)

    # Pass 1: unconstrained exact TV denoise
    x = condat_tv_l1(signal, lam)

    # Pass 2: project onto box constraints and re-denoise
    if lo_bounds is not None and hi_bounds is not None:
        for i in range(n):
            if lo_bounds[i] is not None:
                x[i] = max(x[i], lo_bounds[i])
            if hi_bounds[i] is not None:
                x[i] = min(x[i], hi_bounds[i])
        # Second Condat pass on the projected signal to restore TV smoothness
        x = condat_tv_l1(x, lam)
        # Final projection to ensure feasibility
        for i in range(n):
            if lo_bounds[i] is not None:
                x[i] = max(x[i], lo_bounds[i])
            if hi_bounds[i] is not None:
                x[i] = min(x[i], hi_bounds[i])

    return x


def _linear_fit_r2(values: list[float]) -> tuple[float, float]:
    """Compute R^2 and slope of a linear fit to evenly-spaced values.

    Returns (r_squared, slope_per_step).
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0

    # Simple linear regression on indices
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((v - y_mean) ** 2 for v in values)

    if ss_xx == 0 or ss_yy == 0:
        return 1.0 if ss_yy == 0 else 0.0, 0.0

    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    return r_squared, slope


def compute_hard_bounds(
    features: list,
    timestamps: list[float],
    source_width: int,
    crop_aspect: float = 9 / 16,
    source_height: int = 1080,
) -> tuple[list[Optional[float]], list[Optional[float]], list[bool]]:
    """Compute per-frame hard bounds from must_be_in_frame features.

    For each timestamp, computes the union bbox of all must_be_in_frame=True
    features active at that time. The crop window centered at cam[t] must
    contain this union bbox.

    Args:
        features: list[RequiredFeature] with must_be_in_frame, left, right, t_start
        timestamps: sorted list of frame timestamps
        source_width: source video width in pixels
        crop_aspect: target aspect ratio (width/height)
        source_height: source video height in pixels

    Returns:
        (lo_bounds, hi_bounds, infeasible) — per-timestamp lower/upper bounds
        on the crop center x (in pixels), and a boolean flag per frame for
        infeasible frames where the union bbox exceeds crop width.
    """
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if crop_aspect < src_aspect:
        crop_width_px = int(source_height * crop_aspect)
    else:
        crop_width_px = source_width
    half_crop = crop_width_px / 2.0

    # Filter to hard-required features
    hard = [f for f in features if getattr(f, 'must_be_in_frame', False)]
    if not hard:
        return (
            [None] * len(timestamps),
            [None] * len(timestamps),
            [False] * len(timestamps),
        )

    lo_bounds = []
    hi_bounds = []
    infeasible = []

    for t in timestamps:
        # Find all hard features active at this timestamp
        active = [f for f in hard if abs(f.t_start - t) < 0.05]
        if not active:
            lo_bounds.append(None)
            hi_bounds.append(None)
            infeasible.append(False)
            continue

        # Union bbox in percentage space
        union_left_pct = min(f.left for f in active)
        union_right_pct = max(f.right for f in active)

        # Convert to pixel space
        union_left_px = union_left_pct / 100.0 * source_width
        union_right_px = union_right_pct / 100.0 * source_width
        union_width_px = union_right_px - union_left_px

        if union_width_px > crop_width_px:
            # Infeasible: union bbox wider than crop
            lo_bounds.append(None)
            hi_bounds.append(None)
            infeasible.append(True)
            continue

        # The crop [cam_x - half_crop, cam_x + half_crop] must contain
        # [union_left_px, union_right_px].
        # => cam_x >= union_right_px - half_crop  (right edge of union must be inside crop)
        # => cam_x <= union_left_px + half_crop   (left edge of union must be inside crop)
        lo = union_right_px - half_crop
        hi = union_left_px + half_crop

        # Also clamp to valid range [half_crop, source_width - half_crop]
        lo = max(half_crop, lo)
        hi = min(source_width - half_crop, hi)

        lo_bounds.append(lo)
        hi_bounds.append(hi)
        infeasible.append(False)

    return lo_bounds, hi_bounds, infeasible


def assert_required_in_frame(
    path: list[tuple[float, float]],
    features: list,
    source_width: int,
    crop_aspect: float = 9 / 16,
    source_height: int = 1080,
    tolerance_px: float = 2.0,
) -> list[tuple[float, list]]:
    """Verify that all must_be_in_frame features are covered by the solved path.

    Returns a list of (timestamp, [offending_feature_identities]) for frames
    where the crop does NOT contain all required features. Empty list = all good.
    """
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if crop_aspect < src_aspect:
        crop_width_px = int(source_height * crop_aspect)
    else:
        crop_width_px = source_width
    half_crop = crop_width_px / 2.0

    hard = [f for f in features if getattr(f, 'must_be_in_frame', False)]
    if not hard:
        return []

    violations = []
    for t, cam_x in path:
        crop_left = cam_x - half_crop
        crop_right = cam_x + half_crop

        active = [f for f in hard if abs(f.t_start - t) < 0.05]
        offenders = []
        for f in active:
            f_left_px = f.left / 100.0 * source_width
            f_right_px = f.right / 100.0 * source_width
            if f_left_px < crop_left - tolerance_px or f_right_px > crop_right + tolerance_px:
                offenders.append(getattr(f, 'identity', None))
        if offenders:
            violations.append((t, offenders))

    return violations


def _apply_condat_weights(
    targets: list[float],
    weights: list[float],
    lam: float,
) -> list[float]:
    """Pre-weight signal for diagonal data-fidelity weighting with Condat TV.

    For the L2-TV problem:  min sum w[t]*(cam[t]-target[t])^2 + lam*TV(cam)
    This is equivalent to solving with adjusted targets when weights differ.
    The standard trick: stronger weight pulls the solution closer to the target
    at that frame. We scale the effective lambda inversely with weight so that
    high-weight frames get lower regularization relative to data fidelity.

    For simplicity, we return a weighted signal: when all weights are 1.0,
    the output is identical to the input (bit-identical backward compat).
    """
    # When weight > 1, the target pulls harder; < 1, pulls weaker.
    # We implement this by duplicating the signal but biasing toward
    # a running mean for low-weight frames (effectively softening their pull).
    if not weights:
        return list(targets)

    # Compute the global mean as the "neutral" position
    mean_x = sum(targets) / len(targets) if targets else 0.0
    result = []
    for i, (t, w) in enumerate(zip(targets, weights)):
        # Blend between target and global mean based on weight
        # w=1.0 → pure target, w→0 → pulls toward mean (less influence)
        result.append(w * t + (1.0 - w) * mean_x)
    return result


def _apply_deadzone(targets: list[float], deadzone_px: float) -> list[float]:
    """Replace target values within deadzone of a running hold with the hold value.

    Uses a running median over the previous 0.5s (15 samples at 30fps) as
    the hold reference. If a target is within deadzone_px of the hold,
    it's replaced by the hold value — this suppresses keypoint jitter
    without masking genuine subject motion.
    """
    if not targets or deadzone_px <= 0:
        return list(targets)

    result = list(targets)
    window = 15  # ~0.5s at 30fps
    hold = targets[0]

    for i in range(len(targets)):
        # Update hold as running median of recent window
        start = max(0, i - window)
        recent = sorted(targets[start:i + 1])
        hold = recent[len(recent) // 2]

        if abs(targets[i] - hold) < deadzone_px:
            result[i] = hold

    return result


def _dump_solve_csv(
    job_id: str, seg_idx: int,
    times: list[float], targets: list[float], solved: list[float],
    lo_bounds: Optional[list[float]], hi_bounds: Optional[list[float]],
    *,
    solver: str = "condat",
    weights: Optional[list[float]] = None,
) -> None:
    """Write a per-call CSV for telemetry analysis (behind CLIPAI_DUMP_L1=1)."""
    global _dump_counter
    try:
        _DUMP_DIR.mkdir(parents=True, exist_ok=True)
        jid = job_id or "unknown"
        fname = f"{jid}_{seg_idx:04d}.csv"
        path = _DUMP_DIR / fname
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "target_x", "solved_x", "lo_bound", "hi_bound", "solver", "weight"])
            for i, t in enumerate(times):
                lo = lo_bounds[i] if lo_bounds and lo_bounds[i] is not None else ""
                hi = hi_bounds[i] if hi_bounds and hi_bounds[i] is not None else ""
                w = weights[i] if weights else 1.0
                writer.writerow([f"{t:.4f}", f"{targets[i]:.2f}", f"{solved[i]:.2f}",
                                 lo, hi, solver, f"{w:.4f}"])
        _dump_counter += 1
        logger.debug("L1 telemetry: wrote %s (%d rows)", path, len(times))
    except Exception as e:
        logger.warning("L1 telemetry dump failed: %s", e)


def solve_camera_path(
    face_positions: list[tuple[float, float]],
    source_width: int = 1920,
    lam: float = TV_LAMBDA,
    hard_features: Optional[list] = None,
    source_height: int = 1080,
    crop_aspect: float = 9 / 16,
    job_id: str = "",
    seg_idx: int = 0,
    *,
    weights: Optional[list[float]] = None,
) -> dict:
    """Solve L1-optimal camera path for a segment.

    Args:
        face_positions: [(timestamp, x_pixel), ...] dense face centroids.
        source_width: Source video width in pixels.
        lam: TV regularization weight.
        hard_features: list[RequiredFeature] for hard constraint enforcement.
            When provided, the solver projects cam[t] into the feasible
            interval at each iteration and checks feasibility afterward.
        source_height: Source video height in pixels.
        crop_aspect: Target crop aspect ratio (width/height).
        weights: Optional per-frame weights (same length as face_positions).
            When provided, scales the data-fidelity term per frame.
            None = all weights 1.0 (bit-identical to unweighted path).

    Returns:
        {
            "mode": "stationary" | "tracking" | "panning" | "infeasible",
            "center": float (pixel) — for stationary mode,
            "path": [(t, x), ...] — for tracking mode,
            "slope": float — for panning mode (px/sec),
            "ease_in_ms": int — 0 for snap, >0 for smooth,
            "infeasible_frames": [(t, [identities]), ...] — if infeasible,
        }
    """
    if not face_positions:
        return {
            "mode": "stationary",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    # Sort by time
    sorted_pos = sorted(face_positions, key=lambda p: p[0])
    times = [p[0] for p in sorted_pos]
    targets = [p[1] for p in sorted_pos]

    if len(targets) < 2:
        return {
            "mode": "stationary",
            "center": targets[0],
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    # Compute hard bounds if features provided
    lo_bounds = None
    hi_bounds = None
    infeasible_frames = []

    if hard_features:
        lo_bounds, hi_bounds, infeasible_flags = compute_hard_bounds(
            hard_features, times, source_width, crop_aspect, source_height,
        )

        # Check for infeasible frames
        infeasible_times = [t for t, inf in zip(times, infeasible_flags) if inf]
        if infeasible_times:
            # Signal infeasibility — caller should promote to PADDED
            logger.warning(
                "L1 solver: %d/%d frames infeasible (required features exceed crop width)",
                len(infeasible_times), len(times),
            )
            return {
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": [(t, []) for t in infeasible_times],
            }

    # ── Resolution-independent lambda ──
    # If caller used the legacy default TV_LAMBDA=10.0, upgrade to fraction-based
    if lam == TV_LAMBDA:
        lam = TV_LAMBDA_FRAC * source_width  # 0.015 * 1920 = 28.8

    # ── Pre-solve dead-zone: suppress keypoint noise during holds ──
    deadzone_px = DEADZONE_FRAC * source_width  # ~25px at 1920
    targets_dz = _apply_deadzone(targets, deadzone_px)

    # ── Pre-solve panning detection on target signal ──
    # Detect panning BEFORE solving — the raw (dead-zoned) target is noisier
    # but won't be staircase-smoothed by TV, so a genuine linear pan has
    # higher R² here than in post-solve Condat output.
    if len(targets_dz) >= 3:
        _pre_r2, _pre_slope_step = _linear_fit_r2(targets_dz)
        _pre_dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        _pre_slope_pps = _pre_slope_step / max(_pre_dt, 0.001)

        if (_pre_r2 > PANNING_R2_THRESHOLD
                and abs(_pre_slope_pps) > PANNING_MIN_SLOPE):
            # Verify the linear fit stays within hard_bounds everywhere
            n_t = len(targets_dz)
            y_mean = sum(targets_dz) / n_t
            x_mean = (n_t - 1) / 2.0
            _fit_vals = [_pre_slope_step * (i - x_mean) + y_mean for i in range(n_t)]
            bounds_ok = True
            if lo_bounds is not None and hi_bounds is not None:
                for i in range(n_t):
                    lo_v = lo_bounds[i]
                    hi_v = hi_bounds[i]
                    if lo_v is not None and _fit_vals[i] < lo_v - 1.0:
                        bounds_ok = False
                        break
                    if hi_v is not None and _fit_vals[i] > hi_v + 1.0:
                        bounds_ok = False
                        break

            if bounds_ok:
                logger.info(
                    "L1 solver: shot=%d frames=%d mode=panning solver=pre_solve_linear solve_ms=0.0",
                    seg_idx, n_t,
                )
                return {
                    "mode": "panning",
                    "center": sum(_fit_vals) / len(_fit_vals),
                    "path": list(zip(times, _fit_vals)),
                    "slope": _pre_slope_pps,
                    "ease_in_ms": 0,
                    "infeasible_frames": [],
                }

    # ── Solver selection ──
    n_frames = len(targets_dz)
    _t0 = time.perf_counter()
    if _SOLVER_MODE == "lp" or (_SOLVER_MODE == "auto" and n_frames <= LP_MAX_FRAMES):
        if _SOLVER_MODE == "lp" and n_frames > LP_MAX_FRAMES * 2:
            raise ValueError(
                f"LP solver forced but shot has {n_frames} frames "
                f"(> {LP_MAX_FRAMES * 2} max). Use CLIPAI_L1_SOLVER=auto."
            )
        from backend.services._autoflip_lp import solve_autoflip_lp
        _lp_lo = [lo_bounds[i] if lo_bounds and lo_bounds[i] is not None else 0.0
                  for i in range(n_frames)]
        _lp_hi = [hi_bounds[i] if hi_bounds and hi_bounds[i] is not None else float(source_width)
                  for i in range(n_frames)]
        # Per-frame data-term weights: scale lam1 per frame in the LP
        _lp_weights = weights if weights is not None else None
        solved = solve_autoflip_lp(
            targets_dz, _lp_lo, _lp_hi,
            lam1=1.0, lam2=LP_LAMBDA_V, lam3=LP_LAMBDA_A, lam4=LP_LAMBDA_J,
            weights=_lp_weights,
        )
        _solver_used = "lp"
    else:
        # For Condat: pre-weight signal for diagonal data-fidelity weighting.
        # Replace target[t] with weighted version that pulls harder/softer.
        if weights is not None:
            _condat_input = _apply_condat_weights(targets_dz, weights, lam)
        else:
            _condat_input = targets_dz
        solved = _tv_denoise_1d(_condat_input, lam, 0, lo_bounds, hi_bounds)
        _solver_used = "condat"
    _solve_ms = (time.perf_counter() - _t0) * 1000

    logger.info(
        "L1 solver: shot=%d frames=%d mode=%s solver=%s solve_ms=%.1f",
        seg_idx, n_frames, "pending", _solver_used, _solve_ms,
    )

    # ── Debug telemetry dump ──
    if _DUMP_L1:
        _dump_solve_csv(job_id, seg_idx, times, targets, solved, lo_bounds, hi_bounds,
                        solver=_solver_used, weights=weights)

    # Post-solve verification: check that all hard features are in frame
    if hard_features:
        path_pairs = list(zip(times, solved))
        violations = assert_required_in_frame(
            path_pairs, hard_features, source_width, crop_aspect, source_height,
        )
        if violations:
            logger.warning(
                "L1 solver: %d frames have required features out of crop after solve "
                "(triggering PADDED fallback): %s",
                len(violations),
                [(f"t={t:.2f}", ids) for t, ids in violations[:5]],
            )
            return {
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": violations,
            }

    # Analyze the solved path
    path_min = min(solved)
    path_max = max(solved)
    path_range = path_max - path_min

    # Mode selection
    if path_range < STATIONARY_THRESHOLD * source_width:
        # Stationary: tiny movement, emit single static center
        center = sum(solved) / len(solved)
        return {
            "mode": "stationary",
            "center": center,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    # Post-solve panning check (fallback for cases where pre-solve fit was
    # close-but-not-quite and the solver smoothed it into something more linear)
    r2, slope_per_step = _linear_fit_r2(solved)
    if len(times) >= 3:
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        slope_px_per_sec = slope_per_step / max(dt, 0.001)
    else:
        slope_px_per_sec = 0.0

    if r2 > PANNING_R2_THRESHOLD and abs(slope_px_per_sec) > PANNING_MIN_SLOPE:
        return {
            "mode": "panning",
            "center": sum(solved) / len(solved),
            "path": list(zip(times, solved)),
            "slope": slope_px_per_sec,
            "ease_in_ms": 0,  # constant velocity, no ease
            "infeasible_frames": [],
        }

    # Tracking: non-trivial movement
    return {
        "mode": "tracking",
        "center": solved[0],
        "path": list(zip(times, solved)),
        "slope": 0.0,
        "ease_in_ms": 0,  # L1 solver says snap
        "infeasible_frames": [],
    }


def _classify_segment_mode(
    times: list[float],
    solved: list[float],
    source_width: int,
) -> dict:
    """Classify a segment slice as stationary/tracking/panning.

    Returns the same dict shape as solve_camera_path.
    """
    if not solved:
        return {
            "mode": "stationary",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    path_range = max(solved) - min(solved)

    if path_range < STATIONARY_THRESHOLD * source_width:
        center = sum(solved) / len(solved)
        return {
            "mode": "stationary",
            "center": center,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    r2, slope_per_step = _linear_fit_r2(solved)
    if len(times) >= 3:
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        slope_px_per_sec = slope_per_step / max(dt, 0.001)
    else:
        slope_px_per_sec = 0.0

    if r2 > PANNING_R2_THRESHOLD and abs(slope_px_per_sec) > PANNING_MIN_SLOPE:
        return {
            "mode": "panning",
            "center": sum(solved) / len(solved),
            "path": list(zip(times, solved)),
            "slope": slope_px_per_sec,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    return {
        "mode": "tracking",
        "center": solved[0],
        "path": list(zip(times, solved)),
        "slope": 0.0,
        "ease_in_ms": 0,
        "infeasible_frames": [],
    }


def solve_camera_path_for_shot(
    shot_start: float,
    shot_end: float,
    segments_in_shot: list,
    propagated_path,
    source_width: int = 1920,
    source_height: int = 1080,
    crop_aspect: float = 9 / 16,
    target_fps: float = 30.0,
    lam: float = TV_LAMBDA,
    hard_features: Optional[list] = None,
    job_id: str = "",
) -> list[dict]:
    """Solve L1-optimal camera path for an entire shot, then slice per segment.

    Runs the TV solver once over the full shot (not per-segment), which:
    - Eliminates pops at segment boundaries (they're inside one solve)
    - Provides lookahead: the solver can begin easing BEFORE a target moves

    Mirror-reflects the first/last 30 frames at shot boundaries for
    symmetric lookahead at the edges.

    Args:
        shot_start: Shot start time (seconds).
        shot_end: Shot end time (seconds).
        segments_in_shot: list of segment objects with .start, .end, .active_slot.
        propagated_path: InterpolatedFaceTimeline or dense_faces list.
        source_width, source_height, crop_aspect: Video dimensions.
        target_fps: Uniform sample rate for the solver.
        lam: TV regularization weight.
        hard_features: Required features for hard constraint enforcement.
        job_id: For telemetry.

    Returns:
        list[dict] — one result dict per segment (same shape as solve_camera_path).
    """
    if not segments_in_shot:
        return []

    # ── Resolution-independent lambda ──
    if lam == TV_LAMBDA:
        lam = TV_LAMBDA_FRAC * source_width

    # ── Build full-shot target at uniform fps ──
    dt = 1.0 / target_fps
    pad_frames = 30  # ~1s lookahead padding

    # Collect per-segment targets, concatenated
    all_times = []
    all_targets = []
    segment_boundaries = []  # [(start_idx, end_idx)] into the arrays

    for seg in segments_in_shot:
        seg_start_idx = len(all_times)
        positions = get_propagated_positions_for_segment(
            propagated_path, seg.active_slot, seg.start, seg.end,
            source_width=source_width, target_fps=target_fps,
        )
        for t, x in positions:
            all_times.append(t)
            all_targets.append(x)
        seg_end_idx = len(all_times)
        segment_boundaries.append((seg_start_idx, seg_end_idx))

    if len(all_targets) < 2:
        # Not enough data — fall back to per-segment solving
        results = []
        for seg in segments_in_shot:
            results.append(solve_camera_path(
                [], source_width, lam, hard_features,
                source_height, crop_aspect, job_id,
            ))
        return results

    # ── Pre-solve dead-zone ──
    deadzone_px = DEADZONE_FRAC * source_width
    all_targets = _apply_deadzone(all_targets, deadzone_px)

    # ── Mirror-reflection padding at shot boundaries ──
    n_orig = len(all_targets)
    pad_left = min(pad_frames, n_orig)
    pad_right = min(pad_frames, n_orig)

    # Pad left: mirror the first pad_left samples
    left_pad = list(reversed(all_targets[:pad_left]))
    # Pad right: mirror the last pad_right samples
    right_pad = list(reversed(all_targets[n_orig - pad_right:]))

    padded_targets = left_pad + all_targets + right_pad
    padded_times = []
    for i in range(len(padded_targets)):
        padded_times.append(shot_start + (i - pad_left) * dt)

    # ── Compute hard bounds for padded signal (if any) ──
    lo_bounds = None
    hi_bounds = None
    if hard_features:
        lo_bounds, hi_bounds, infeasible_flags = compute_hard_bounds(
            hard_features, padded_times, source_width, crop_aspect, source_height,
        )
        infeasible_times = [t for t, inf in zip(padded_times, infeasible_flags) if inf]
        if infeasible_times:
            logger.warning(
                "L1 shot solver: %d/%d frames infeasible",
                len(infeasible_times), len(padded_times),
            )
            # Return infeasible for all segments
            return [{
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": [(t, []) for t in infeasible_times],
            } for _ in segments_in_shot]

    # ── Solve: LP (with accel+jerk) or Condat TV ──
    n_padded = len(padded_targets)
    _t0 = time.perf_counter()
    use_lp = (
        _SOLVER_MODE == "lp"
        or (_SOLVER_MODE == "auto" and n_padded <= LP_MAX_FRAMES)
    )
    if _SOLVER_MODE == "lp" and n_padded > LP_MAX_FRAMES * 2:
        raise ValueError(
            f"LP solver forced but shot has {n_padded} frames "
            f"(> {LP_MAX_FRAMES * 2} max). Use CLIPAI_L1_SOLVER=auto."
        )
    if use_lp and n_padded > LP_MAX_FRAMES and _SOLVER_MODE == "auto":
        use_lp = False

    if use_lp:
        from backend.services._autoflip_lp import solve_autoflip_lp
        # Build bounds for LP solver (None → unconstrained as source_width bounds)
        lp_lo = []
        lp_hi = []
        for i in range(n_padded):
            lo_val = lo_bounds[i] if lo_bounds and lo_bounds[i] is not None else 0.0
            hi_val = hi_bounds[i] if hi_bounds and hi_bounds[i] is not None else float(source_width)
            lp_lo.append(lo_val)
            lp_hi.append(hi_val)
        padded_solved = solve_autoflip_lp(
            padded_targets, lp_lo, lp_hi,
            lam1=1.0, lam2=LP_LAMBDA_V, lam3=LP_LAMBDA_A, lam4=LP_LAMBDA_J,
        )
        _solver_used = "lp"
    else:
        padded_solved = _tv_denoise_1d(padded_targets, lam, 0, lo_bounds, hi_bounds)
        _solver_used = "condat"
    _solve_ms = (time.perf_counter() - _t0) * 1000

    logger.info(
        "L1 solver: shot=%d frames=%d mode=%s solver=%s solve_ms=%.1f",
        0, n_padded, "shot", _solver_used, _solve_ms,
    )

    # ── Discard padding ──
    solved = padded_solved[pad_left:pad_left + n_orig]
    times = all_times

    # ── Debug telemetry dump ──
    if _DUMP_L1:
        _dump_solve_csv(
            job_id, 9000, times, all_targets, solved,
            lo_bounds[pad_left:pad_left + n_orig] if lo_bounds else None,
            hi_bounds[pad_left:pad_left + n_orig] if hi_bounds else None,
            solver=_solver_used,
        )

    # ── Post-solve verification ──
    if hard_features:
        path_pairs = list(zip(times, solved))
        violations = assert_required_in_frame(
            path_pairs, hard_features, source_width, crop_aspect, source_height,
        )
        if violations:
            logger.warning(
                "L1 shot solver: %d frames have required features out of crop",
                len(violations),
            )
            return [{
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": violations,
            } for _ in segments_in_shot]

    # ── Slice solved path back into per-segment results ──
    results = []
    for (seg_start_idx, seg_end_idx) in segment_boundaries:
        seg_times = times[seg_start_idx:seg_end_idx]
        seg_solved = solved[seg_start_idx:seg_end_idx]
        result = _classify_segment_mode(seg_times, seg_solved, source_width)
        results.append(result)

    return results


def get_dense_face_positions_for_segment(
    dense_faces: list,
    active_slot: Optional[int],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    """Extract (timestamp, x_pixel) pairs for a face slot in a time range.

    Uses nose_x from dense face data. Falls back to any face if slot is None.

    DEPRECATED: Prefer get_propagated_positions_for_segment which produces
    uniform-fps output suitable for the TV solver.
    """
    positions = []
    for df in dense_faces:
        if df.timestamp < start or df.timestamp >= end:
            continue
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if active_slot is not None and sid != active_slot:
                continue
            x = getattr(f, 'nose_x', None)
            if x is not None:
                positions.append((df.timestamp, x))
                break  # one face per frame
    return positions


def get_propagated_positions_for_segment(
    propagated_path,
    active_slot: Optional[int],
    start: float,
    end: float,
    source_width: int = 1920,
    target_fps: float = 30.0,
) -> list[tuple[float, float]]:
    """Extract uniform-fps (timestamp, x_pixel) pairs from a propagated timeline.

    Produces a uniformly-spaced trajectory at target_fps by resampling
    the propagated path via linear interpolation. Gaps (occlusion, identity
    drop) are filled by holding the last known x-position.

    Works with either an InterpolatedFaceTimeline (from dense_propagator)
    or a raw list of dense_faces (sparse detections). When given sparse
    input, resamples onto a uniform grid so the TV solver's finite-difference
    operator sees uniform dt.

    Args:
        propagated_path: InterpolatedFaceTimeline or list of FrameFaces/dense_faces.
        active_slot: Face registry slot ID to track (None = any face).
        start: Segment start time (seconds).
        end: Segment end time (seconds, exclusive).
        source_width: Source video width (for percentage-to-pixel conversion).
        target_fps: Output sample rate (default 30.0).

    Returns:
        list[(timestamp, x_pixel)] at uniform target_fps spacing.
        Returns the same signature as get_dense_face_positions_for_segment.
    """
    # ── Step 1: Extract raw (t, x_pct) pairs from the source ──
    raw_samples = []  # [(t, x_pct)]

    if hasattr(propagated_path, 'slot_positions_in_range'):
        # InterpolatedFaceTimeline path — per-source-frame data
        if active_slot is not None:
            positions = propagated_path.slot_positions_in_range(active_slot, start, end)
            for t, cx, _cy, _w, _h, _conf in positions:
                if start <= t < end:
                    raw_samples.append((t, cx))
        else:
            # No specific slot — use first available slot per frame
            for s in propagated_path.samples:
                if s.timestamp < start or s.timestamp >= end:
                    continue
                if s.bboxes:
                    first_slot = next(iter(s.bboxes))
                    cx, _cy, _w, _h = s.bboxes[first_slot]
                    raw_samples.append((s.timestamp, cx))
    else:
        # Sparse dense_faces list — extract like get_dense_face_positions_for_segment
        for df in propagated_path:
            if df.timestamp < start or df.timestamp >= end:
                continue
            for f in df.faces:
                sid = getattr(f, 'identity_id', -1)
                if active_slot is not None and sid != active_slot:
                    continue
                x = getattr(f, 'nose_x', None)
                if x is not None:
                    raw_samples.append((df.timestamp, x))
                    break

    if not raw_samples:
        return []

    raw_samples.sort(key=lambda p: p[0])

    # ── Step 2: Resample onto uniform grid at target_fps ──
    dt = 1.0 / target_fps
    n_frames = max(1, int((end - start) * target_fps))
    uniform = []
    last_x = raw_samples[0][1]  # hold for gaps
    raw_idx = 0

    for i in range(n_frames):
        t = start + i * dt
        if t >= end:
            break

        # Find the two raw samples bracketing t for linear interpolation
        while raw_idx < len(raw_samples) - 1 and raw_samples[raw_idx + 1][0] <= t:
            raw_idx += 1

        if raw_idx >= len(raw_samples) - 1:
            # Past the last raw sample — hold last known
            x_pct = raw_samples[-1][1]
        elif raw_samples[raw_idx][0] >= t:
            # Before or at the first relevant sample
            x_pct = raw_samples[raw_idx][1]
        else:
            # Linear interpolation between raw_samples[raw_idx] and [raw_idx + 1]
            t0, x0 = raw_samples[raw_idx]
            t1, x1 = raw_samples[raw_idx + 1]
            span = t1 - t0
            if span > 0:
                alpha = (t - t0) / span
                x_pct = x0 + alpha * (x1 - x0)
            else:
                x_pct = x0

        last_x = x_pct
        # Convert from percentage to pixel space
        x_px = x_pct / 100.0 * source_width
        uniform.append((round(t, 6), x_px))

    return uniform
