import numpy as np
import scipy.sparse as sparse
import osqp


def compute_contact_frame(n, c, Jr):
    """
    Compute the contact frame.
    @param[in] n: (N, P, 3)
    @param[in] c: (N, P, 3)
    @param[in] Jr: (N, P, 3, nq)
    @return: Jo: (N, P, 3, 6)
    @return: Jr_contact: (N, P, 3, nq)
    @return: n: (N, P, 3)
    @return: t1: (N, P, 3)
    @return: t2: (N, P, 3)
    """
    N = n.shape[0]
    P = n.shape[1]

    n = n / np.linalg.norm(n, axis=-1, keepdims=True)  # (N, P, 3)
    # pick a reference axis that is not parallel to n
    e1 = np.zeros_like(n)
    e1[..., 0] = 1.0
    e2 = np.zeros_like(n)
    e2[..., 1] = 1.0

    # if |n·e1| is too close to 1, use e2, else use e1
    use_e2 = np.abs(np.sum(n * e1, axis=-1, keepdims=True)) > 0.9
    ref = np.where(use_e2, e2, e1)

    t1 = np.cross(ref, n, axis=-1)
    t1 = t1 / np.linalg.norm(t1, axis=-1, keepdims=True)

    t2 = np.cross(n, t1, axis=-1)
    t2 = t2 / np.linalg.norm(t2, axis=-1, keepdims=True)

    # checks
    assert np.max(np.abs(np.sum(n * t1, axis=-1))) < 1e-6
    assert np.max(np.abs(np.sum(n * t2, axis=-1))) < 1e-6
    assert np.max(np.abs(np.sum(t1 * t2, axis=-1))) < 1e-6
    assert np.max(np.abs(np.cross(t1, t2, axis=-1) - n)) < 1e-6

    Jr_n = (Jr * n.reshape(N, P, 3, 1)).sum(axis=-2, keepdims=True)  # (N, P, 1, nq)
    Jr_t1 = (Jr * t1.reshape(N, P, 3, 1)).sum(axis=-2, keepdims=True)  # (N, P, 1, nq)
    Jr_t2 = (Jr * t2.reshape(N, P, 3, 1)).sum(axis=-2, keepdims=True)  # (N, P, 1, nq)
    Jr_contact = np.concatenate([Jr_n, Jr_t1, Jr_t2], axis=-2)  # (N, P, 3, nq)

    Jo = np.zeros((N, P, 3, 6))
    Jo[:, :, 0, :3] = n
    Jo[:, :, 1, :3] = t1
    Jo[:, :, 2, :3] = t2
    Jo[:, :, 0, 3:] = np.cross(c, n, axis=-1)
    Jo[:, :, 1, 3:] = np.cross(c, t1, axis=-1)
    Jo[:, :, 2, 3:] = np.cross(c, t2, axis=-1)
    return Jo, Jr_contact, n, t1, t2


def generate_contact_forces_legacy(
    Jo, Jr_contact, torque_limit, min_fn, mu, null_space_margin=0.01,
    enable_torque_limit=False, enable_achievable_force=False,
    gravity_wrench=None,
):
    """
    Generate contact forces for a given grasp pose.
    @param[in] Jo: (P, 3, 6)
    @param[in] Jr_contact: (P, 3, nq)
    @param[in] torque_limit: (nq,)
    @param[in] min_fn: scalar
    @param[in] mu: scalar
    @param[in] null_space_margin: scalar
    @param[in] gravity_wrench: optional (6,) external wrench acting on the
        object (e.g. gravity at the cad-frame origin) in the SAME frame as
        ``Jo``. When provided, the cost becomes ``||G f + gravity_wrench||²``
        so the contact net wrench is driven to ``-gravity_wrench`` instead of
        zero. Defaults to ``None`` (no external wrench, original behavior).
    @return: f: (P, 3)
    """
    P = Jo.shape[0]  # number of contact points
    nq = Jr_contact.shape[2]  # number of joints

    # === QP problem ===
    m = osqp.OSQP()
    Jo_flat = Jo.reshape(-1, 6)
    P_np = Jo_flat @ Jo_flat.T  # (3P, 3P)
    P_sparse = sparse.csc_matrix(P_np)

    # Linear cost term: 0 for pure force closure, G^T @ gravity_wrench when
    # compensating an external wrench (½||G f + g_w||² expansion).
    if gravity_wrench is None:
        q = np.zeros((3 * P))
    else:
        q = Jo_flat @ np.asarray(gravity_wrench, dtype=float)

    list_A = []
    list_l = []
    list_u = []

    # # 0. Static equilibrium: net wrench on object must be zero (net force + net torque = 0)
    # # net_wrench = sum_p Jo_i[p].T @ f_p, so G @ f_flat = net_wrench with G (6, 3*P)
    # G = np.hstack([Jo_i[p].T for p in range(P)])  # (6, 3*P)
    # list_A.append(sparse.csc_matrix(G))
    # list_l.append(np.zeros(6, dtype=float))
    # list_u.append(np.zeros(6, dtype=float))

    # Main object constraint
    # 1. torque limit
    if enable_torque_limit:
        list_A.append(sparse.csc_matrix(Jr_contact.reshape(P * 3, nq).T))  # (nq, 3P)
        list_l.append(-torque_limit.astype(float))  # (nq,)
        list_u.append(torque_limit.astype(float))  # (nq,)

    # 2. minimum total normal force: sum fn >= min_fn_i
    cols = 3 * np.arange(P)
    rows = np.zeros(P, dtype=int)
    data = np.ones(P)
    A_min_fn = sparse.csc_matrix((data, (rows, cols)), shape=(1, 3 * P))

    list_A.append(A_min_fn)  # (1, 3P)
    list_l.append(np.array([min_fn], dtype=float))  # (1,)
    list_u.append(np.array([np.inf], dtype=float))  # (1,)

    # 3. positive normal force: fn_i >= 0
    rows = np.arange(P)
    cols = 3 * rows
    data = np.ones(P)
    A_pos_fn = sparse.csc_matrix((data, (rows, cols)), shape=(P, 3 * P))

    list_A.append(A_pos_fn)  # (P, 3P)
    list_l.append(np.zeros(P, dtype=float))  # (P,)
    list_u.append(np.full(P, np.inf, dtype=float))  # (P,)

    # 4. friction cone (Linear friction pyramid)
    # |ft1| + |ft2| \leq mu * fn
    # contact indices
    k = np.arange(P)

    fn = 3 * k + 0
    ft1 = 3 * k + 1
    ft2 = 3 * k + 2

    # 4 inequalities per contact
    rows = np.repeat(4 * k, 3)
    rows = np.concatenate([rows + 0, rows + 1, rows + 2, rows + 3])

    cols = np.concatenate(
        [
            np.stack([fn, ft1, ft2], axis=1),
            np.stack([fn, ft1, ft2], axis=1),
            np.stack([fn, ft1, ft2], axis=1),
            np.stack([fn, ft1, ft2], axis=1),
        ]
    ).ravel()

    data = np.concatenate(
        [
            np.tile([-mu, +1.0, +1.0], P),  # +ft1 +ft2 - mu fn <= 0
            np.tile([-mu, +1.0, -1.0], P),  # +ft1 -ft2 - mu fn <= 0
            np.tile([-mu, -1.0, +1.0], P),  # -ft1 +ft2 - mu fn <= 0
            np.tile([-mu, -1.0, -1.0], P),  # -ft1 -ft2 - mu fn <= 0
        ]
    )

    A_fric = sparse.csc_matrix((data, (rows, cols)), shape=(4 * P, 3 * P))

    list_A.append(A_fric)  # (4P, 3P)
    list_l.append(-np.inf * np.ones(4 * P))  # (4P,)
    list_u.append(np.zeros(4 * P))  # (4P,)

    # 5. Achievable contact force: f \in Range(J_r_contact)
    if enable_achievable_force:
        # (I - J(J^TJ)^{-1}J^T) f = 0
        Jr_contact_flat = Jr_contact.reshape(P * 3, nq)  # (3P, nq)
        I = np.eye(P * 3)  # (3P, 3P)
        JtJ = Jr_contact_flat.T @ Jr_contact_flat  # (nq, nq)
        JtJ_inv = np.linalg.inv(JtJ + 1e-6 * np.eye(nq))  # (nq, nq)
        I_minus_JtJ_invJ = I - Jr_contact_flat @ JtJ_inv @ Jr_contact_flat.T  # (3P, 3P)
        A_achievable = sparse.csc_matrix(I_minus_JtJ_invJ)
        list_A.append(A_achievable)
        list_l.append(-null_space_margin * np.ones(3 * P))
        list_u.append(null_space_margin * np.ones(3 * P))

    # concat all constraints
    A_sparse = sparse.vstack(list_A, format="csc")
    l = np.hstack(list_l)
    u = np.hstack(list_u)

    assert A_sparse.shape[0] == l.size == u.size
    assert A_sparse.shape[1] == 3 * P

    m.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, verbose=False,
            max_iter=1000, polish=False)
    res = m.solve()

    solved = res.info.status_val in (1, 2)  # 1=solved, 2=solved inaccurate
    if not solved:
        print(
            f"OSQP failed: status={res.info.status_val}, "
            f"status_str={res.info.status}"
        )
        return np.zeros((P, 3)), solved

    f = res.x.reshape(P, 3)

    net_wrench = (
        (np.transpose(Jo, (0, 2, 1)) @ f.reshape(P, 3, 1)).reshape(P, 6).sum(axis=0)
    )  # (6,) = [force (3), torque (3)]
    # print(f"net_force: {net_wrench[:3]} N")
    # print(f"net_torque: {net_wrench[3:]} Nm")

    fn = f[:, 0]  # (P,)
    # print(f"fn: {fn} N")
    return f, solved

def generate_contact_forces(
    Jo, Jr_contact, torque_limit, min_fn, mu, null_space_margin=0.01,
    enable_torque_limit=False, enable_achievable_force=False,
    gravity_wrench=None, actuated_mask=None,
    contact_epsilon=None, eps_weight=1.0,
    verbose=True,
):
    """OSQP force-closure QP with rank-6 lifting w = G f.

    contact_epsilon: optional (P,) — adds (eps_weight/2) Σ ε_i² ‖f_i‖² to cost.
    eps_weight: scalar λ scaling the robust term. None/0 ε ⇒ nominal.
    """
    P = Jo.shape[0]
    nq = Jr_contact.shape[2]
    n_f = 3 * P
    n_w = 6
    n_var = n_f + n_w  # decision variable z = [f; w]

    # --- Hessian: blkdiag(λ·diag(ε_i² I₃), I₆) ---
    if contact_epsilon is None:
        f_block = sparse.csc_matrix((n_f, n_f))
    else:
        eps = np.asarray(contact_epsilon, dtype=float)
        assert eps.shape == (P,), f"contact_epsilon must be (P,), got {eps.shape}"
        f_block = sparse.diags(
            float(eps_weight) * np.repeat(eps ** 2, 3), format="csc",
        )
    P_sparse = sparse.block_diag(
        [f_block, sparse.eye(n_w, format="csc")], format="csc",
    )

    # --- Linear cost ---
    # original: ½ f^T G^T G f + (G^T g_w)^T f  =  ½||w||² + g_w^T w
    if gravity_wrench is None:
        q = np.zeros(n_var)
    else:
        g_w = np.asarray(gravity_wrench, dtype=float)
        q = np.concatenate([np.zeros(n_f), g_w])

    # --- helper: pad (m, n_f) constraint → (m, n_var) with zero w-columns ---
    def pad(A_f):
        return sparse.hstack(
            [A_f, sparse.csc_matrix((A_f.shape[0], n_w))], format="csc"
        )

    list_A = []
    list_l = []
    list_u = []

    # 0. Coupling  w = G f  ⟹  [G, −I_6] z = 0
    G = Jo.reshape(-1, 6).T  # (6, 3P)
    A_coupling = sparse.hstack(
        [sparse.csc_matrix(G), -sparse.eye(n_w, format="csc")], format="csc"
    )
    list_A.append(A_coupling)
    list_l.append(np.zeros(n_w))
    list_u.append(np.zeros(n_w))

    # 1. torque limit
    if enable_torque_limit:
        A_torque = sparse.csc_matrix(Jr_contact.reshape(n_f, nq).T)
        list_A.append(pad(A_torque))
        list_l.append(-torque_limit.astype(float))
        list_u.append(torque_limit.astype(float))

    # 2. minimum total normal force
    cols = 3 * np.arange(P)
    A_min_fn = sparse.csc_matrix(
        (np.ones(P), (np.zeros(P, dtype=int), cols)), shape=(1, n_f)
    )
    list_A.append(pad(A_min_fn))
    list_l.append(np.array([min_fn], dtype=float))
    list_u.append(np.array([np.inf], dtype=float))

    # 3. positive normal force
    rows = np.arange(P)
    cols = 3 * rows
    A_pos_fn = sparse.csc_matrix(
        (np.ones(P), (rows, cols)), shape=(P, n_f)
    )
    list_A.append(pad(A_pos_fn))
    list_l.append(np.zeros(P, dtype=float))
    list_u.append(np.full(P, np.inf, dtype=float))

    # 4. friction cone (linear pyramid)
    k = np.arange(P)
    fn_idx = 3 * k
    ft1_idx = 3 * k + 1
    ft2_idx = 3 * k + 2

    rows = np.repeat(4 * k, 3)
    rows = np.concatenate([rows + 0, rows + 1, rows + 2, rows + 3])
    cols = np.concatenate([
        np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
        np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
        np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
        np.stack([fn_idx, ft1_idx, ft2_idx], axis=1),
    ]).ravel()
    data = np.concatenate([
        np.tile([-mu, +1.0, +1.0], P),
        np.tile([-mu, +1.0, -1.0], P),
        np.tile([-mu, -1.0, +1.0], P),
        np.tile([-mu, -1.0, -1.0], P),
    ])
    A_fric = sparse.csc_matrix((data, (rows, cols)), shape=(4 * P, n_f))
    list_A.append(pad(A_fric))
    list_l.append(-np.inf * np.ones(4 * P))
    list_u.append(np.zeros(4 * P))

    # 5. achievable contact force (only for actuated contacts)
    if enable_achievable_force:
        # Select only the actuated contact rows (3 rows per contact)
        if actuated_mask is not None:
            act_rows = np.concatenate([3 * np.where(actuated_mask)[0] + d for d in range(3)])
            act_rows.sort()
        else:
            act_rows = np.arange(n_f)
        Jr_act = Jr_contact.reshape(n_f, nq)[act_rows]  # (3*P_act, nq)

        # Faster: extract only the null-space basis vectors via SVD.
        # N^T f_act ≈ 0 constrains actuated contacts to lie in Range(Jr).
        U, s, _ = np.linalg.svd(Jr_act, full_matrices=True)
        rank = int(np.sum(s > 1e-6))
        N = U[:, rank:]  # (3*P_act, 3*P_act - rank) null-space basis
        if N.shape[1] > 0:
            # Map back to full f vector: constraint only touches actuated rows
            N_full = np.zeros((N.shape[1], n_f))
            N_full[:, act_rows] = N.T
            list_A.append(pad(sparse.csc_matrix(N_full)))
            list_l.append(-null_space_margin * np.ones(N.shape[1]))
            list_u.append(null_space_margin * np.ones(N.shape[1]))

    # --- assemble & solve ---
    A_sparse = sparse.vstack(list_A, format="csc")
    l = np.hstack(list_l)
    u = np.hstack(list_u)

    m = osqp.OSQP()
    m.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u,
            verbose=False, max_iter=1000, polish=False)
    res = m.solve()

    solved = res.info.status_val in (1, 2)
    if not solved:
        if verbose:
            print(
                f"OSQP failed: status={res.info.status_val}, "
                f"status_str={res.info.status}"
            )
        return np.zeros((P, 3)), solved

    f = res.x[:n_f].reshape(P, 3)
    return f, solved