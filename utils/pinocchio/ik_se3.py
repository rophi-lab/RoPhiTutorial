"""Damped SE(3) CLIK for a Pinocchio frame (arm joints free, others locked)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pinocchio as pin


def solve_frame_ik_se3(
    model: pin.Model,
    data: pin.Data,
    q_seed: np.ndarray,
    T_des: np.ndarray,
    frame_id: int,
    *,
    free_joint_idx: Optional[Sequence[int]] = None,
    max_iters: int = 200,
    tol: float = 1e-4,
    damp: float = 1e-4,
    step: float = 0.5,
    q_lower: Optional[np.ndarray] = None,
    q_upper: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], float, bool]:
    """Solve frame SE(3) IK with damped least-squares (CLIK).

    Error (Pinocchio body / LOCAL layout ``(ν, ω)``)::

        ξ = log(T^{-1} T_des)^∨

    Only indices in ``free_joint_idx`` are updated (default: all ``nq``).

    Returns
    -------
    q_sol : ndarray or None
        Solution configuration (copy), or ``None`` if failed.
    err_norm : float
        Final ``||ξ||``.
    success : bool
        ``True`` if ``err_norm < tol`` and within joint limits.
    """
    q = np.asarray(q_seed, dtype=np.float64).copy()
    nq = model.nq
    if free_joint_idx is None:
        free = np.arange(nq, dtype=int)
    else:
        free = np.asarray(free_joint_idx, dtype=int)

    oMd = pin.SE3(np.asarray(T_des[:3, :3], dtype=np.float64), T_des[:3, 3])
    if q_lower is None:
        q_lower = model.lowerPositionLimit
    if q_upper is None:
        q_upper = model.upperPositionLimit

    err_norm = np.inf
    for _ in range(int(max_iters)):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[frame_id]
        xi = pin.log6(oMf.actInv(oMd)).vector  # (v, w)
        err_norm = float(np.linalg.norm(xi))
        if err_norm < tol:
            break

        J_full = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL
        )  # 6 x nq
        J = J_full[:, free]
        # Damped least squares: dq_f = J^T (J J^T + λ I)^{-1} ξ
        JJT = J @ J.T
        lam = float(damp) * float(np.trace(JJT) / max(JJT.shape[0], 1)) + 1e-8
        dq_f = J.T @ np.linalg.solve(JJT + lam * np.eye(6), xi)
        q[free] = q[free] + float(step) * dq_f
        q = np.clip(q, q_lower, q_upper)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    xi = pin.log6(data.oMf[frame_id].actInv(oMd)).vector
    err_norm = float(np.linalg.norm(xi))
    in_limits = bool(np.all(q >= q_lower - 1e-9) and np.all(q <= q_upper + 1e-9))
    success = err_norm < tol and in_limits
    return (q if success else None), err_norm, success


def random_so3(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    """Random rotation near identity: ``exp([ω]_×)`` with ``||ω|| ≤ max_angle``."""
    if max_angle <= 0:
        return np.eye(3)
    w = rng.normal(size=3)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        return np.eye(3)
    w = w / n * float(rng.uniform(0.0, max_angle))
    return pin.exp3(w)


def make_SE3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T
