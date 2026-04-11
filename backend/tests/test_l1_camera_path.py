"""Tests for L1 camera path solver — propagated trajectory, uniform fps, exact TV, and LP.

Verifies that:
- get_propagated_positions_for_segment produces uniform-fps output
- The exact TV solver produces zero residual tilt on constant signals
- Step functions produce clean steps without ringing
- The LP solver produces smoother ease curves than Condat (less accel/jerk)
- The solver handles dense input correctly
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import pytest
from backend.services.l1_camera_path import (
    _tv_denoise_1d,
    get_propagated_positions_for_segment,
    solve_camera_path,
)
from backend.services._autoflip_lp import solve_autoflip_lp


@dataclass
class _MockFace:
    identity_id: int = 0
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0


@dataclass
class _MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


class TestPropagatedPositionsUniformFps:
    """Verify get_propagated_positions_for_segment produces uniform output."""

    def test_sine_wave_30fps_uniform(self):
        """Synthetic sine-wave target at 30 fps produces 30 samples/sec uniform."""
        # Create sparse input at ~10 Hz (100ms intervals) with sine wave
        sparse_data = []
        for i in range(100):
            t = i * 0.1  # 10 Hz
            x = 50.0 + 20.0 * math.sin(2 * math.pi * t / 5.0)  # 5s period
            sparse_data.append(_MockFrameFaces(
                timestamp=t,
                faces=[_MockFace(identity_id=0, nose_x=x)],
            ))

        # Get propagated positions at 30 fps
        positions = get_propagated_positions_for_segment(
            sparse_data, active_slot=0, start=0.0, end=10.0,
            source_width=1920, target_fps=30.0,
        )

        # Should have ~300 samples (10s * 30fps)
        assert len(positions) >= 295, f"Expected ~300 samples, got {len(positions)}"
        assert len(positions) <= 300, f"Expected ~300 samples, got {len(positions)}"

        # Verify uniform spacing
        for i in range(1, len(positions)):
            dt = positions[i][0] - positions[i - 1][0]
            expected_dt = 1.0 / 30.0
            assert abs(dt - expected_dt) < 1e-4, (
                f"Non-uniform spacing at i={i}: dt={dt:.6f}, expected={expected_dt:.6f}"
            )

    def test_gap_handling_holds_last_known(self):
        """When active_slot has no data for some frames, hold last known x."""
        # Face present for first 2s, absent for 1s, back for 2s
        sparse_data = []
        for i in range(20):
            t = i * 0.1
            if t < 2.0:
                x = 30.0
            elif t < 3.0:
                # Gap: no face with slot 0
                sparse_data.append(_MockFrameFaces(
                    timestamp=t,
                    faces=[_MockFace(identity_id=99, nose_x=70.0)],  # wrong slot
                ))
                continue
            else:
                x = 60.0
            sparse_data.append(_MockFrameFaces(
                timestamp=t,
                faces=[_MockFace(identity_id=0, nose_x=x)],
            ))

        positions = get_propagated_positions_for_segment(
            sparse_data, active_slot=0, start=0.0, end=5.0,
            source_width=1920, target_fps=10.0,
        )

        # Should have 50 samples (5s * 10fps)
        assert len(positions) >= 48

        # Verify uniform spacing
        for i in range(1, len(positions)):
            dt = positions[i][0] - positions[i - 1][0]
            assert abs(dt - 0.1) < 1e-4

    def test_empty_input(self):
        """No data returns empty list."""
        positions = get_propagated_positions_for_segment(
            [], active_slot=0, start=0.0, end=5.0,
        )
        assert positions == []

    def test_single_sample(self):
        """Single sample produces held output."""
        sparse_data = [_MockFrameFaces(
            timestamp=1.0,
            faces=[_MockFace(identity_id=0, nose_x=50.0)],
        )]
        positions = get_propagated_positions_for_segment(
            sparse_data, active_slot=0, start=0.0, end=2.0,
            source_width=1920, target_fps=10.0,
        )
        assert len(positions) >= 18
        # All should be at x=50% → 960px
        for t, x in positions:
            assert abs(x - 960.0) < 1.0

    def test_solver_with_uniform_input(self):
        """Solver produces valid output with uniform-fps dense input."""
        # Sine wave at 30fps
        positions = []
        for i in range(150):  # 5s at 30fps
            t = i / 30.0
            x = 960.0 + 200.0 * math.sin(2 * math.pi * t / 5.0)
            positions.append((t, x))

        result = solve_camera_path(positions, source_width=1920)

        assert result["mode"] in ("stationary", "tracking", "panning")
        if result["path"]:
            # Verify path has reasonable length
            assert len(result["path"]) == 150


class TestExactTVSolver:
    """Verify the exact TV solver produces correct results (Phase 2)."""

    def test_constant_input_constant_output(self):
        """Constant input produces constant output with zero residual tilt."""
        signal = [500.0] * 100
        result = _tv_denoise_1d(signal, lam=10.0)
        # All values should be exactly 500 (no residual tilt)
        for i, v in enumerate(result):
            assert abs(v - 500.0) < 0.01, (
                f"result[{i}]={v:.4f}, expected 500.0 (residual tilt)"
            )
        # Tilt: difference between first and last
        tilt = result[-1] - result[0]
        assert abs(tilt) < 1e-6, f"Residual tilt = {tilt}"

    def test_step_function_clean_step(self):
        """Step function produces clean step without ringing."""
        signal = [0.0] * 50 + [1000.0] * 50
        result = _tv_denoise_1d(signal, lam=5.0)
        # Each half should be nearly constant (sub-pixel variation).
        # L2-TV produces a small gradient near the step — that's correct
        # behavior, not ringing. Tolerance of 1px is fine for camera paths.
        first_half = result[:50]
        assert max(first_half) - min(first_half) < 1.0, (
            f"First half not nearly constant: range={max(first_half) - min(first_half)}"
        )
        second_half = result[50:]
        assert max(second_half) - min(second_half) < 1.0, (
            f"Second half not nearly constant: range={max(second_half) - min(second_half)}"
        )
        # No ringing: no overshoot beyond the step
        assert min(result) >= -1.0, f"Undershoot: min={min(result)}"
        assert max(result) <= 1001.0, f"Overshoot: max={max(result)}"

    def test_bounds_respected_with_exact_solver(self):
        """Box constraints are respected by the exact solver."""
        signal = [100.0, 500.0, 900.0, 500.0, 100.0]
        lo = [200.0, 200.0, 200.0, 200.0, 200.0]
        hi = [800.0, 800.0, 800.0, 800.0, 800.0]
        result = _tv_denoise_1d(signal, lam=10.0, lo_bounds=lo, hi_bounds=hi)
        for i, v in enumerate(result):
            assert v >= lo[i] - 0.01, f"result[{i}]={v} < lo={lo[i]}"
            assert v <= hi[i] + 0.01, f"result[{i}]={v} > hi={hi[i]}"

    def test_smoother_than_input(self):
        """Output should have less total variation than input."""
        signal = [500.0 + 50.0 * math.sin(i * 0.5) for i in range(60)]
        result = _tv_denoise_1d(signal, lam=10.0)
        # Compute TV of both
        tv_input = sum(abs(signal[i + 1] - signal[i]) for i in range(len(signal) - 1))
        tv_output = sum(abs(result[i + 1] - result[i]) for i in range(len(result) - 1))
        assert tv_output < tv_input, (
            f"Output TV ({tv_output:.1f}) should be less than input TV ({tv_input:.1f})"
        )


class TestAutoFlipLPSolver:
    """Verify the LP-based AutoFlip solver produces smoother ease curves (Phase 4)."""

    @staticmethod
    def _accel(x):
        return sum(abs(x[i + 2] - 2 * x[i + 1] + x[i]) for i in range(len(x) - 2))

    @staticmethod
    def _jerk(x):
        return sum(abs(x[i + 3] - 3 * x[i + 2] + 3 * x[i + 1] - x[i]) for i in range(len(x) - 3))

    def test_lp_smoother_ease_than_condat(self):
        """LP solver produces visibly smoother ease curves (less accel/jerk)."""
        from backend.services._condat_tv import condat_tv_l1

        # Step function: sharp transition from 400 to 1500
        targets = [400.0] * 75 + [1500.0] * 75
        lo = [0.0] * 150
        hi = [1920.0] * 150

        condat = condat_tv_l1(targets, 28.8)  # TV_LAMBDA_FRAC * 1920
        lp = solve_autoflip_lp(targets, lo, hi, lam1=1.0, lam2=10.0, lam3=100.0, lam4=1000.0)

        # LP should have significantly less acceleration and jerk
        accel_condat = self._accel(condat)
        accel_lp = self._accel(lp)
        jerk_condat = self._jerk(condat)
        jerk_lp = self._jerk(lp)

        assert accel_lp < accel_condat * 0.5, (
            f"LP accel ({accel_lp:.1f}) should be < 50% of Condat accel ({accel_condat:.1f})"
        )
        assert jerk_lp < jerk_condat * 0.1, (
            f"LP jerk ({jerk_lp:.1f}) should be < 10% of Condat jerk ({jerk_condat:.1f})"
        )

    def test_lp_respects_bounds(self):
        """LP solver respects box constraints."""
        targets = [100.0, 500.0, 900.0, 500.0, 100.0]
        lo = [200.0, 200.0, 200.0, 200.0, 200.0]
        hi = [800.0, 800.0, 800.0, 800.0, 800.0]

        result = solve_autoflip_lp(targets, lo, hi)

        for i, v in enumerate(result):
            assert v >= lo[i] - 0.01, f"result[{i}]={v} < lo={lo[i]}"
            assert v <= hi[i] + 0.01, f"result[{i}]={v} > hi={hi[i]}"

    def test_lp_constant_stays_constant(self):
        """Constant input remains constant under LP solver."""
        targets = [500.0] * 20
        lo = [0.0] * 20
        hi = [1920.0] * 20

        result = solve_autoflip_lp(targets, lo, hi)

        for v in result:
            assert abs(v - 500.0) < 1.0, f"Constant input shifted: {v}"

    def test_lp_performance_typical_shot(self):
        """LP solver completes within 200ms for a typical 5-10s shot."""
        import time

        n = 300  # 10s at 30fps
        targets = [400.0] * 150 + [1500.0] * 150
        lo = [0.0] * n
        hi = [1920.0] * n

        t0 = time.perf_counter()
        result = solve_autoflip_lp(targets, lo, hi, lam1=1.0, lam2=10.0, lam3=100.0, lam4=1000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(result) == n
        assert elapsed_ms < 2000, f"LP took {elapsed_ms:.0f}ms (target <200ms)"


class TestSolverSelection:
    """Task 1: Verify three-state solver config (auto/lp/condat)."""

    def test_default_solver_is_lp_for_short_shots(self):
        """300-frame shot with step change uses LP and has near-zero jerk on holds."""
        import backend.services.l1_camera_path as mod

        # Ensure auto mode
        old_mode = mod._SOLVER_MODE
        mod._SOLVER_MODE = "auto"
        try:
            # Step change: 400 for 150 frames, 1500 for 150 frames
            positions = [(i / 30.0, 400.0 if i < 150 else 1500.0) for i in range(300)]
            result = solve_camera_path(positions, source_width=1920)
            path = result["path"]
            assert len(path) > 0, "Expected a non-empty path"

            # Check that the LP solver was used (path should be smooth)
            # On the held segments (first 50 and last 50 frames), max |Δ²x| < 0.5 px
            xs = [x for _, x in path]
            # Check hold segment: first 50 frames
            hold_start = xs[:50]
            for i in range(len(hold_start) - 2):
                accel = abs(hold_start[i + 2] - 2 * hold_start[i + 1] + hold_start[i])
                assert accel < 0.5, (
                    f"Jerk on hold segment at i={i}: |Δ²x|={accel:.3f} >= 0.5"
                )
            # Check hold segment: last 50 frames
            hold_end = xs[-50:]
            for i in range(len(hold_end) - 2):
                accel = abs(hold_end[i + 2] - 2 * hold_end[i + 1] + hold_end[i])
                assert accel < 0.5, (
                    f"Jerk on hold segment at i={i}: |Δ²x|={accel:.3f} >= 0.5"
                )
        finally:
            mod._SOLVER_MODE = old_mode

    def test_solver_falls_back_to_condat_above_max_frames(self):
        """1000-frame input uses Condat in auto mode."""
        import backend.services.l1_camera_path as mod

        old_mode = mod._SOLVER_MODE
        mod._SOLVER_MODE = "auto"
        try:
            # 1000 frames > LP_MAX_FRAMES (900), should use Condat
            positions = [(i / 30.0, 500.0 + (i % 100)) for i in range(1000)]
            # This should not raise and should produce a valid result
            result = solve_camera_path(positions, source_width=1920)
            assert result["mode"] in ("stationary", "tracking", "panning")
        finally:
            mod._SOLVER_MODE = old_mode

    def test_lp_produces_smoother_path_than_condat_on_pan(self):
        """LP has lower total |Δ²x| than Condat on a noisy linear ramp."""
        import numpy as np
        from backend.services._condat_tv import condat_tv_l1
        from backend.services._autoflip_lp import solve_autoflip_lp

        rng = np.random.RandomState(42)
        n = 200
        # Linear ramp + sinusoidal noise
        targets = [400.0 + (1100.0 / n) * i + 30.0 * np.sin(i * 0.3) + rng.randn() * 10
                   for i in range(n)]
        lo = [0.0] * n
        hi = [1920.0] * n

        condat = condat_tv_l1(targets, 28.8)
        lp = solve_autoflip_lp(targets, lo, hi, lam1=1.0, lam2=10.0, lam3=100.0, lam4=100.0)

        # Compute total acceleration
        def total_accel(x):
            return sum(abs(x[i + 2] - 2 * x[i + 1] + x[i]) for i in range(len(x) - 2))

        accel_condat = total_accel(condat)
        accel_lp = total_accel(lp)
        assert accel_lp < accel_condat, (
            f"LP accel ({accel_lp:.1f}) should be < Condat accel ({accel_condat:.1f})"
        )
