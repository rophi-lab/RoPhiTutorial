"""Sequential QP smoothing of multi-finger tip paths (ICRA TO reaching).

After a heuristic via-point initialization, refine interior waypoints with a
few SQP iterations that (1) minimize discrete path curvature / jerk-like
second differences and (2) enforce linearized clearance to an object
distance field. Matches the two-finger OSQP loop in Manipulator-Software
``ReactiveGrasping._construct_fingertip_velocity_fields`` (type=\"TO\").
"""

from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse import vstack


def smooth_multi_finger_paths_sqp(
    paths: np.ndarray,
    dist_grad_func,
    num_iterations: int = 3,
    constraints_margin: float = 0.01,
    other_dist_grad_funcs=None,
    other_constraints_margin: float = 0.05,
    savgol_window: int = 5,
    savgol_poly: int = 2,
) -> np.ndarray:
    """Smooth fingertip paths with SQP + optional Savitzky–Golay filter.

    Parameters
    ----------
    paths :
        ``(F, T, 3)`` trajectories including endpoints (T = N_interior + 2).
    dist_grad_func :
        Callable ``(N, 3) -> (dists (N,), grads (N, 3))`` for the grasp object.
    num_iterations :
        Number of SQP linearization / OSQP solves (paper default: 3).
    constraints_margin :
        Require ``dist(x) >= margin`` on interior waypoints (grasp object).
    other_dist_grad_funcs :
        Optional list of callables with the same signature for moving obstacles.
    other_constraints_margin :
        Clearance margin for those obstacles.
    savgol_window, savgol_poly :
        Post-SQP temporal smoothing (set window <= 1 to skip).

    Returns
    -------
    paths_out : ``(F, T, 3)``
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != 3:
        raise ValueError(f"paths must be (F, T, 3), got {paths.shape}")

    F, T, _ = paths.shape
    if T < 3:
        return paths.copy()

    n_int = T - 2  # interior knots per finger
    if n_int < 1:
        return paths.copy()

    # Re-pin endpoints from the input (heuristic may already have them).
    trajs = [paths[f].copy() for f in range(F)]

    I3 = np.eye(3)
    main_diag = 2.0 * np.ones(n_int)
    off_diag = -1.0 * np.ones(n_int - 1) if n_int > 1 else np.array([])
    if n_int == 1:
        P_1D = sparse.csc_matrix([[2.0]])
    else:
        P_1D = sparse.diags(
            diagonals=[off_diag, main_diag, off_diag],
            offsets=[-1, 0, 1],
            format="csc",
        )
    P_block = sparse.kron(P_1D, sparse.csc_matrix(I3))
    P = sparse.block_diag([P_block] * F, format="csc")

    n_wp = F * n_int  # number of 3D interior waypoints
    n_var = 3 * n_wp
    row_idx = np.repeat(np.arange(n_wp), 3)
    col_idx = np.arange(n_var)
    other_funcs = list(other_dist_grad_funcs or [])

    for _ in range(int(num_iterations)):
        # Linear cost of ½||D x||² around current: g = Dᵀ D x_cur for interior.
        g_parts = []
        for f in range(F):
            xf = trajs[f]
            # (N, 3): 2 x_i - x_{i-1} - x_{i+1} for interior i
            gf = 2.0 * xf[1:-1] - xf[:-2] - xf[2:]
            g_parts.append(gf.reshape(-1))
        g = np.concatenate(g_parts)

        x_int = np.concatenate([trajs[f][1:-1] for f in range(F)], axis=0)  # (F*N, 3)
        list_A = []
        list_l = []
        list_u = []

        dist, grad = dist_grad_func(x_int)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        grad = np.asarray(grad, dtype=np.float64).reshape(-1, 3)
        if dist.shape[0] != n_wp or grad.shape != (n_wp, 3):
            raise ValueError(
                f"dist_grad_func returned shapes {dist.shape}, {grad.shape}; "
                f"expected ({n_wp},), ({n_wp}, 3)"
            )
        list_A.append(
            sparse.coo_matrix(
                (grad.reshape(-1), (row_idx, col_idx)), shape=(n_wp, n_var)
            ).tocsc()
        )
        list_l.append(constraints_margin - dist)
        list_u.append(np.full(n_wp, 1000.0, dtype=np.float64))

        for other_fn in other_funcs:
            d_o, g_o = other_fn(x_int)
            d_o = np.asarray(d_o, dtype=np.float64).reshape(-1)
            g_o = np.asarray(g_o, dtype=np.float64).reshape(-1, 3)
            list_A.append(
                sparse.coo_matrix(
                    (g_o.reshape(-1), (row_idx, col_idx)), shape=(n_wp, n_var)
                ).tocsc()
            )
            list_l.append(other_constraints_margin - d_o)
            list_u.append(np.full(n_wp, 1000.0, dtype=np.float64))

        A = vstack(list_A)
        l = np.hstack(list_l)
        u = np.hstack(list_u)

        m = osqp.OSQP()
        m.setup(P=P, q=g, A=A, l=l, u=u, verbose=False)
        res = m.solve()
        if res.info.status_val not in (
            osqp.constant("OSQP_SOLVED"),
            osqp.constant("OSQP_SOLVED_INACCURATE"),
        ):
            # Keep current iterate if this SQP step fails.
            continue

        dx = res.x.reshape(F, n_int, 3)
        for f in range(F):
            trajs[f][1:-1] += dx[f]

    # Temporal smoothing (same as reference).
    if savgol_window is not None and int(savgol_window) >= 3:
        w = int(savgol_window)
        if w % 2 == 0:
            w += 1
        w = min(w, T if T % 2 == 1 else T - 1)
        if w >= 3 and w > int(savgol_poly):
            for f in range(F):
                trajs[f] = savgol_filter(
                    trajs[f],
                    window_length=w,
                    polyorder=int(savgol_poly),
                    mode="interp",
                    axis=0,
                )
            # Re-pin endpoints after filter drift.
            for f in range(F):
                trajs[f][0] = paths[f, 0]
                trajs[f][-1] = paths[f, -1]

    return np.stack(trajs, axis=0)
