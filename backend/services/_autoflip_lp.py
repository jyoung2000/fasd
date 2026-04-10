"""LP-based AutoFlip camera path solver with acceleration and jerk penalties.

Formulates the AutoFlip objective (Grundmann et al. 2011, Section 4.2) as a
linear program using scipy's HiGHS backend:

    min  lam1 * sum |cam[t] - target[t]|
       + lam2 * sum |cam[t] - cam[t-1]|         (velocity)
       + lam3 * sum |cam[t] - 2*cam[t-1] + cam[t-2]|  (acceleration)
       + lam4 * sum |cam[t] - 3*cam[t-1] + 3*cam[t-2] - cam[t-3]|  (jerk)
    s.t. lo[t] <= cam[t] <= hi[t]

Each |·| term is linearized via an auxiliary slack variable and two inequality
constraints: |z| <= s  <=>  z <= s, -z <= s, s >= 0.

Reference:
    M. Grundmann, V. Kwatra, and I. Essa, "Auto-Directed Video Stabilization
    with Robust L1 Optimal Camera Paths," CVPR 2011, Section 4.2.
"""

import logging

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

logger = logging.getLogger(__name__)


def solve_autoflip_lp(
    targets: list[float],
    lo_bounds: list[float],
    hi_bounds: list[float],
    lam1: float = 1.0,
    lam2: float = 10.0,
    lam3: float = 100.0,
    lam4: float = 1000.0,
) -> list[float]:
    """Solve the AutoFlip L1-optimal camera path via linear programming.

    Args:
        targets: Target face positions (pixels), length n.
        lo_bounds: Per-frame lower bounds on camera position.
        hi_bounds: Per-frame upper bounds on camera position.
        lam1: Data fidelity weight.
        lam2: Velocity smoothness weight.
        lam3: Acceleration smoothness weight.
        lam4: Jerk smoothness weight.

    Returns:
        Solved camera path, length n.
    """
    n = len(targets)
    if n <= 1:
        return list(targets)

    # Variable layout:
    #   cam[0..n-1]       — camera positions (n vars)
    #   s1[0..n-1]        — |cam[t] - target[t]| slack (n vars)
    #   s2[0..n-2]        — |cam[t+1] - cam[t]| slack (n-1 vars)
    #   s3[0..n-3]        — |accel| slack (n-2 vars)
    #   s4[0..n-4]        — |jerk| slack (n-3 vars)
    #
    # Total variables: n + n + (n-1) + (n-2) + (n-3) = 5n - 6

    n_s2 = max(n - 1, 0)
    n_s3 = max(n - 2, 0)
    n_s4 = max(n - 3, 0)
    n_vars = n + n + n_s2 + n_s3 + n_s4

    # Offsets into the variable vector
    off_cam = 0
    off_s1 = n
    off_s2 = off_s1 + n
    off_s3 = off_s2 + n_s2
    off_s4 = off_s3 + n_s3

    # ── Objective: min lam1*sum(s1) + lam2*sum(s2) + lam3*sum(s3) + lam4*sum(s4)
    c = np.zeros(n_vars)
    c[off_s1:off_s1 + n] = lam1
    c[off_s2:off_s2 + n_s2] = lam2
    c[off_s3:off_s3 + n_s3] = lam3
    c[off_s4:off_s4 + n_s4] = lam4

    # ── Inequality constraints: A_ub @ x <= b_ub ──
    # Each |z| <= s  =>  z - s <= 0  AND  -z - s <= 0
    # So 2 rows per |·| term.
    n_constraints = 2 * n + 2 * n_s2 + 2 * n_s3 + 2 * n_s4
    A = lil_matrix((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    row = 0

    # s1: |cam[t] - target[t]| <= s1[t]
    #   cam[t] - s1[t] <= target[t]
    #   -cam[t] - s1[t] <= -target[t]
    for t in range(n):
        A[row, off_cam + t] = 1.0
        A[row, off_s1 + t] = -1.0
        b[row] = targets[t]
        row += 1
        A[row, off_cam + t] = -1.0
        A[row, off_s1 + t] = -1.0
        b[row] = -targets[t]
        row += 1

    # s2: |cam[t+1] - cam[t]| <= s2[t]
    #   cam[t+1] - cam[t] - s2[t] <= 0
    #   cam[t] - cam[t+1] - s2[t] <= 0
    for t in range(n_s2):
        A[row, off_cam + t + 1] = 1.0
        A[row, off_cam + t] = -1.0
        A[row, off_s2 + t] = -1.0
        b[row] = 0.0
        row += 1
        A[row, off_cam + t] = 1.0
        A[row, off_cam + t + 1] = -1.0
        A[row, off_s2 + t] = -1.0
        b[row] = 0.0
        row += 1

    # s3: |cam[t+2] - 2*cam[t+1] + cam[t]| <= s3[t]
    for t in range(n_s3):
        # cam[t+2] - 2*cam[t+1] + cam[t] - s3[t] <= 0
        A[row, off_cam + t + 2] = 1.0
        A[row, off_cam + t + 1] = -2.0
        A[row, off_cam + t] = 1.0
        A[row, off_s3 + t] = -1.0
        b[row] = 0.0
        row += 1
        # -(cam[t+2] - 2*cam[t+1] + cam[t]) - s3[t] <= 0
        A[row, off_cam + t + 2] = -1.0
        A[row, off_cam + t + 1] = 2.0
        A[row, off_cam + t] = -1.0
        A[row, off_s3 + t] = -1.0
        b[row] = 0.0
        row += 1

    # s4: |cam[t+3] - 3*cam[t+2] + 3*cam[t+1] - cam[t]| <= s4[t]
    for t in range(n_s4):
        # cam[t+3] - 3*cam[t+2] + 3*cam[t+1] - cam[t] - s4[t] <= 0
        A[row, off_cam + t + 3] = 1.0
        A[row, off_cam + t + 2] = -3.0
        A[row, off_cam + t + 1] = 3.0
        A[row, off_cam + t] = -1.0
        A[row, off_s4 + t] = -1.0
        b[row] = 0.0
        row += 1
        # -(cam[t+3] - 3*cam[t+2] + 3*cam[t+1] - cam[t]) - s4[t] <= 0
        A[row, off_cam + t + 3] = -1.0
        A[row, off_cam + t + 2] = 3.0
        A[row, off_cam + t + 1] = -3.0
        A[row, off_cam + t] = 1.0
        A[row, off_s4 + t] = -1.0
        b[row] = 0.0
        row += 1

    A = A.tocsc()

    # ── Variable bounds ──
    bounds = []
    for t in range(n):
        bounds.append((lo_bounds[t], hi_bounds[t]))  # cam[t]
    for _ in range(n):
        bounds.append((0.0, None))  # s1[t] >= 0
    for _ in range(n_s2):
        bounds.append((0.0, None))  # s2[t] >= 0
    for _ in range(n_s3):
        bounds.append((0.0, None))  # s3[t] >= 0
    for _ in range(n_s4):
        bounds.append((0.0, None))  # s4[t] >= 0

    # ── Solve ──
    result = linprog(
        c, A_ub=A, b_ub=b, bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 10.0},
    )

    if not result.success:
        logger.warning("AutoFlip LP solver failed: %s — falling back to targets", result.message)
        return list(targets)

    cam = result.x[off_cam:off_cam + n].tolist()
    return cam
