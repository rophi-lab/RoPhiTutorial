"""Force-closure contact-force QP (OSQP).

Ported from Manipulator-Software
``scripts/planning/offline/grasp_squeeze_params_generator/force_gen.py``.
"""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sparse


def compute_contact_frame(n, c, Jr):
    """Build contact frames (n, t1, t2) and wrench/Jacobian maps.

    Parameters
    ----------
    n : (N, P, 3)
    c : (N, P, 3)
    Jr : (N, P, 3, nq)

    Returns
    -------
    Jo : (N, P, 3, 6)
    Jr_contact : (N, P, 3, nq)
    n, t1, t2 : (N, P, 3)
    """
    N = n.shape[0]
    P = n.shape[1]

    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    e1 = np.zeros_like(n)
    e1[..., 0] = 1.0
    e2 = np.zeros_like(n)
    e2[..., 1] = 1.0
    use_e2 = np.abs(np.sum(n * e1, axis=-1, keepdims=True)) > 0.9
    ref = np.where(use_e2, e2, e1)

    t1 = np.cross(ref, n, axis=-1)
    t1 = t1 / np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(n, t1, axis=-1)
    t2 = t2 / np.linalg.norm(t2, axis=-1, keepdims=True)

    Jr_n = (Jr * n.reshape(N, P, 3, 1)).sum(axis=-2, keepdims=True)
    Jr_t1 = (Jr * t1.reshape(N, P, 3, 1)).sum(axis=-2, keepdims=True)
    Jr_t2 = (Jr * t2.reshape(N, P, 3, 1)).sum(axis=-2, keepdims=True)
    Jr_contact = np.concatenate([Jr_n, Jr_t1, Jr_t2], axis=-2)

    Jo = np.zeros((N, P, 3, 6))
    Jo[:, :, 0, :3] = n
    Jo[:, :, 1, :3] = t1
    Jo[:, :, 2, :3] = t2
    Jo[:, :, 0, 3:] = np.cross(c, n, axis=-1)
    Jo[:, :, 1, 3:] = np.cross(c, t1, axis=-1)
    Jo[:, :, 2, 3:] = np.cross(c, t2, axis=-1)
    return Jo, Jr_contact, n, t1, t2


def generate_contact_forces(
    Jo,
    Jr_contact,
    torque_limit,
    min_fn,
    mu,
    null_space_margin=0.01,
    enable_torque_limit=False,
    enable_achievable_force=False,
    gravity_wrench=None,
    actuated_mask=None,
):
    """Solve force-closure QP for per-contact forces in the (n, t1, t2) basis.

    Uses variable lifting ``w = G f`` so the Hessian stays sparse.

    Parameters
    ----------
    Jo : (P, 3, 6)
    Jr_contact : (P, 3, nq)
    torque_limit : (nq,)
    actuated_mask : optional (P,) bool — passive contacts (e.g. palm) skipped
        in the achievable-force nullspace constraint.

    Returns
    -------
    f : (P, 3)  — [fn, ft1, ft2]
    solved : bool
    """
    P = Jo.shape[0]
    nq = Jr_contact.shape[2]
    n_f = 3 * P
    n_w = 6
    n_var = n_f + n_w

    P_sparse = sparse.block_diag(
        [sparse.csc_matrix((n_f, n_f)), sparse.eye(n_w, format="csc")],
        format="csc",
    )

    if gravity_wrench is None:
        q = np.zeros(n_var)
    else:
        q = np.concatenate([np.zeros(n_f), np.asarray(gravity_wrench, dtype=float)])

    def pad(A_f):
        return sparse.hstack(
            [A_f, sparse.csc_matrix((A_f.shape[0], n_w))], format="csc"
        )

    list_A, list_l, list_u = [], [], []

    G = Jo.reshape(-1, 6).T  # (6, 3P)
    A_coupling = sparse.hstack(
        [sparse.csc_matrix(G), -sparse.eye(n_w, format="csc")], format="csc"
    )
    list_A.append(A_coupling)
    list_l.append(np.zeros(n_w))
    list_u.append(np.zeros(n_w))

    if enable_torque_limit:
        A_torque = sparse.csc_matrix(Jr_contact.reshape(n_f, nq).T)
        list_A.append(pad(A_torque))
        list_l.append(-torque_limit.astype(float))
        list_u.append(torque_limit.astype(float))

    cols = 3 * np.arange(P)
    A_min_fn = sparse.csc_matrix(
        (np.ones(P), (np.zeros(P, dtype=int), cols)), shape=(1, n_f)
    )
    list_A.append(pad(A_min_fn))
    list_l.append(np.array([min_fn], dtype=float))
    list_u.append(np.array([np.inf], dtype=float))

    rows = np.arange(P)
    cols = 3 * rows
    A_pos_fn = sparse.csc_matrix((np.ones(P), (rows, cols)), shape=(P, n_f))
    list_A.append(pad(A_pos_fn))
    list_l.append(np.zeros(P, dtype=float))
    list_u.append(np.full(P, np.inf, dtype=float))

    k = np.arange(P)
    fn_idx, ft1_idx, ft2_idx = 3 * k, 3 * k + 1, 3 * k + 2
    rows = np.repeat(4 * k, 3)
    rows = np.concatenate([rows + 0, rows + 1, rows + 2, rows + 3])
    cols = np.concatenate(
        [
            np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
            np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
            np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
            np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
        ]
    ).ravel()
    data = np.concatenate(
        [
            np.tile([-mu, +1.0, +1.0], P),
            np.tile([-mu, +1.0, -1.0], P),
            np.tile([-mu, -1.0, +1.0], P),
            np.tile([-mu, -1.0, -1.0], P),
        ]
    )
    A_fric = sparse.csc_matrix((data, (rows, cols)), shape=(4 * P, n_f))
    list_A.append(pad(A_fric))
    list_l.append(-np.inf * np.ones(4 * P))
    list_u.append(np.zeros(4 * P))

    if enable_achievable_force:
        if actuated_mask is not None:
            act_rows = np.concatenate(
                [3 * np.where(actuated_mask)[0] + d for d in range(3)]
            )
            act_rows.sort()
        else:
            act_rows = np.arange(n_f)
        Jr_act = Jr_contact.reshape(n_f, nq)[act_rows]
        U, s, _ = np.linalg.svd(Jr_act, full_matrices=True)
        rank = int(np.sum(s > 1e-6))
        N = U[:, rank:]
        if N.shape[1] > 0:
            N_full = np.zeros((N.shape[1], n_f))
            N_full[:, act_rows] = N.T
            list_A.append(pad(sparse.csc_matrix(N_full)))
            list_l.append(-null_space_margin * np.ones(N.shape[1]))
            list_u.append(null_space_margin * np.ones(N.shape[1]))

    A_sparse = sparse.vstack(list_A, format="csc")
    l = np.hstack(list_l)
    u = np.hstack(list_u)

    m = osqp.OSQP()
    m.setup(
        P=P_sparse, q=q, A=A_sparse, l=l, u=u, verbose=False, max_iter=1000, polish=False
    )
    res = m.solve()

    solved = res.info.status_val in (1, 2)
    if not solved:
        print(
            f"[force_gen] OSQP failed: status={res.info.status_val} "
            f"({res.info.status})"
        )
        return np.zeros((P, 3)), solved

    return res.x[:n_f].reshape(P, 3), solved
