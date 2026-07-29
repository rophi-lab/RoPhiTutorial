import numpy as np


def path_to_min_jerk_trajectory(
    q_path: np.ndarray,
    T: float,
    dt: float,
    eps: float = 1e-12,
):
    """
    Convert a joint-space path q_path (N x n) into a time-parameterized trajectory
    with a minimum-jerk speed profile (smooth accel/decel).

    Returns:
      t:   (M,)
      q:   (M, n)
      qd:  (M, n)
      qdd: (M, n)
    """
    q_path = np.asarray(q_path, dtype=float)
    assert q_path.ndim == 2
    N, n = q_path.shape
    assert N >= 2
    assert T > 0 and dt > 0

    # --- 1) arc-length parameter s over the discrete path ---
    dq = np.diff(q_path, axis=0)  # (N-1, n)
    ds = np.linalg.norm(dq, axis=1)  # (N-1,)
    s = np.concatenate(([0.0], np.cumsum(ds)))  # (N,)
    L = s[-1]

    if L < eps:
        # Path is (almost) constant
        t = np.arange(0.0, T + 0.5 * dt, dt)
        q = np.repeat(q_path[:1], repeats=t.size, axis=0)
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        return t, q, qd, qdd

    # --- 2) min-jerk time scaling sigma(t) ---
    t = np.arange(0.0, T + 0.5 * dt, dt)  # (M,)
    tau = np.clip(t / T, 0.0, 1.0)  # (M,)

    sigma = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    dsigma_dt = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    d2sigma_dt2 = (60 * tau - 180 * tau**2 + 120 * tau**3) / (T**2)

    s_t = sigma * L
    sdot_t = dsigma_dt * L
    sddot_t = d2sigma_dt2 * L

    # --- 3) resample q(s) by linear interpolation along s ---
    q = np.empty((t.size, n))
    for j in range(n):
        q[:, j] = np.interp(s_t, s, q_path[:, j])

    # --- 4) approximate dq/ds and d2q/ds2 along the original path, then chain rule ---
    # dq/ds at nodes (N, n)
    dqds = np.zeros((N, n))
    for i in range(1, N - 1):
        denom = s[i + 1] - s[i - 1]
        if denom > eps:
            dqds[i] = (q_path[i + 1] - q_path[i - 1]) / denom
    # one-sided ends
    if s[1] - s[0] > eps:
        dqds[0] = (q_path[1] - q_path[0]) / (s[1] - s[0])
    if s[-1] - s[-2] > eps:
        dqds[-1] = (q_path[-1] - q_path[-2]) / (s[-1] - s[-2])

    # d2q/ds2 at nodes (N, n)
    d2qds2 = np.zeros((N, n))
    for i in range(1, N - 1):
        ds1 = s[i] - s[i - 1]
        ds2 = s[i + 1] - s[i]
        if ds1 > eps and ds2 > eps:
            # non-uniform second derivative approximation
            d2qds2[i] = (
                2.0
                * (
                    (q_path[i + 1] - q_path[i]) / ds2
                    - (q_path[i] - q_path[i - 1]) / ds1
                )
                / (ds1 + ds2)
            )

    # interpolate derivatives to s(t)
    qd = np.empty_like(q)
    qdd = np.empty_like(q)
    for j in range(n):
        dqds_t = np.interp(s_t, s, dqds[:, j])
        d2qds2_t = np.interp(s_t, s, d2qds2[:, j])

        # chain rule:
        # qdot = (dq/ds) * sdot
        # qddot = (d2q/ds2) * sdot^2 + (dq/ds) * sddot
        qd[:, j] = dqds_t * sdot_t
        qdd[:, j] = d2qds2_t * (sdot_t**2) + dqds_t * sddot_t

    return t, q, qd, qdd
