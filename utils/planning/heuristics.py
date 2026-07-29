import numpy as np


import numpy as np


def _as_F_by_xdim(x, x_dim, F=None):
    x = np.asarray(x)

    # Case: flattened (F*x_dim,) -> (F, x_dim)
    if x.ndim == 1:
        if x.shape[0] == x_dim:
            return x[None, :]  # (1, x_dim)

        if x.shape[0] % x_dim == 0:
            F_infer = x.shape[0] // x_dim
            if (F is not None) and (F_infer != F):
                raise ValueError(
                    f"Flattened length implies F={F_infer}, expected F={F}"
                )
            return x.reshape(F_infer, x_dim)

        raise ValueError(
            f"Expected length {x_dim} or multiple of it, got shape {x.shape}"
        )

    # Case: (F, x_dim)
    if x.ndim == 2:
        if x.shape[1] != x_dim:
            raise ValueError(f"Expected shape (F,{x_dim}), got {x.shape}")
        if (F is not None) and (x.shape[0] != F):
            raise ValueError(f"Expected F={F}, got {x.shape[0]}")
        return x

    # Case: (F, T, x_dim) -> take first waypoint (or last, depending on use)
    if x.ndim == 3:
        if x.shape[-1] != x_dim:
            raise ValueError(f"Expected last dim {x_dim}, got {x.shape}")
        if (F is not None) and (x.shape[0] != F):
            raise ValueError(f"Expected F={F}, got {x.shape[0]}")
        return x[:, 0, :]  # or x[:, -1, :] if this is a goal trajectory

    raise ValueError(f"Unsupported shape {x.shape} for x with x_dim={x_dim}")


def heuristic_path_initialization_general(
    x_ft_init,
    x_ft_goal,
    x_dim,
    num_samples,
    traj_length,
    length,
    constraints_margin,
    dist_func,
):
    """
    Generate collision-heuristic initial trajectories for general number of fingers
    via sampling equally spaced directions on S² or S¹.

    Args:
        x_ft_init: (F, x_dim)
        x_ft_goal: (F, x_dim)
        x_dim: 2 or 3
        num_samples: number of sampled directions
        traj_length: number of internal knots (output length is traj_length+2 incl endpoints)
        length: via-point offset magnitude
        constraints_margin: signed distance threshold (violation if dist < margin)
        dist_func: callable accepting (N, x_dim) points -> (N,) signed distances

    Returns:
        x_ft_traj: (F, traj_length+2, x_dim) trajectory for each finger
    """
    x_ft_init = _as_F_by_xdim(x_ft_init, x_dim)
    x_ft_goal = _as_F_by_xdim(x_ft_goal, x_dim)

    if x_ft_init.shape != x_ft_goal.shape:
        raise ValueError(
            f"init/goal shape mismatch: {x_ft_init.shape} vs {x_ft_goal.shape}"
        )

    F = x_ft_init.shape[0]

    assert x_ft_init.shape[1] == x_dim
    assert x_dim in (2, 3)

    T_out = traj_length + 2  # include endpoints

    def uniform_unit_vectors_2d(n):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def fibonacci_sphere(n):
        i = np.arange(n)
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.stack([x, y, z], axis=1)

    def constant_speed_2segment(x0, x1, x2, n):
        """Return n points sampled uniformly along line x0 → x1 → x2."""
        l1 = np.linalg.norm(x1 - x0)
        l2 = np.linalg.norm(x2 - x1)
        total = l1 + l2

        # Handle degenerate cases robustly
        if total < 1e-12:
            return np.repeat(x0[None, :], n, axis=0)
        if l1 < 1e-12:
            # all on second segment
            t = np.linspace(0, 1, n)[:, None]
            return x1 + (x2 - x1) * t
        if l2 < 1e-12:
            # all on first segment
            t = np.linspace(0, 1, n)[:, None]
            return x0 + (x1 - x0) * t

        t = np.linspace(0, 1, n)  # fraction of total arclength
        split = l1 / total
        pts = np.empty((n, x0.shape[0]), dtype=np.float64)

        mask1 = t <= split
        mask2 = ~mask1

        # segment 1
        tau1 = (t[mask1] * total) / l1
        pts[mask1] = x0 + (x1 - x0) * tau1[:, None]

        # segment 2
        tau2 = ((t[mask2] - split) * total) / l2
        pts[mask2] = x1 + (x2 - x1) * tau2[:, None]

        return pts

    # --- 1) Check linear trajectory for all fingers ---
    t = np.linspace(0, 1, T_out).reshape(1, -1, 1)  # (1, T, 1)
    x_ft_linear_traj = (1 - t) * x_ft_init[:, None, :] + t * x_ft_goal[
        :, None, :
    ]  # (F, T, x_dim)

    dists = dist_func(x_ft_linear_traj.reshape(-1, x_dim)).reshape(F, T_out)
    num_violations_linear = (dists < constraints_margin).sum(axis=1)  # (F,)

    if num_violations_linear.sum() == 0:
        return x_ft_linear_traj

    # --- 2) Sample directions and build via candidates (shared direction set, per-finger via point) ---
    u = (
        uniform_unit_vectors_2d(num_samples)
        if x_dim == 2
        else fibonacci_sphere(num_samples)
    )  # (S, x_dim)
    # via points are offset from each finger's goal
    x_ft_via = x_ft_goal[None, :, :] + u[:, None, :] * length  # (S, F, x_dim)

    # Build 2-segment polyline candidates with half time for each segment (like your 2-finger code)
    T_half = traj_length // 2 + 1  # includes endpoints per half
    t_half = np.linspace(0, 1, T_half).reshape(1, 1, -1, 1)  # (1,1,Th,1)

    x0 = x_ft_init.reshape(1, F, 1, x_dim)  # (1, F, 1, x_dim)
    x2 = x_ft_goal.reshape(1, F, 1, x_dim)  # (1, F, 1, x_dim)
    x1 = x_ft_via[:, :, None, :]  # (S, F, 1, x_dim)

    ft_traj_candidates = np.concatenate(
        [(1 - t_half) * x0 + t_half * x1, (1 - t_half) * x1 + t_half * x2],
        axis=2,
    )  # (S, F, 2*Th, x_dim)

    # Note: this yields length 2*Th, which equals traj_length+2 when traj_length is even.
    # If traj_length is odd, 2*Th = traj_length+3; we'll resample to T_out later anyway.

    # Evaluate signed distances: shape (S, F, Tcand)
    S, _, Tcand, _ = ft_traj_candidates.shape
    ft_dists = dist_func(ft_traj_candidates.reshape(-1, x_dim)).reshape(S, F, Tcand)

    num_violations = (ft_dists < constraints_margin).sum(axis=2)  # (S, F)
    min_violations = num_violations.min(axis=0)  # (F,)

    # --- 3) For each finger, pick best candidate independently (min violations, tie-break by shortest length) ---
    x_ft_traj = np.empty((F, T_out, x_dim), dtype=np.float64)

    for f in range(F):
        best_idxs = np.where(num_violations[:, f] == min_violations[f])[0]

        # tie-break: pick minimal polyline length (init->via->goal)
        x0_f = x_ft_init[f]
        x2_f = x_ft_goal[f]
        via_fs = x_ft_via[best_idxs, f, :]  # (K, x_dim)
        lengths = np.linalg.norm(via_fs - x0_f[None, :], axis=1) + np.linalg.norm(
            x2_f[None, :] - via_fs, axis=1
        )

        best_idx = best_idxs[np.argmin(lengths)]
        x1_best = x_ft_via[best_idx, f, :]

        # constant-speed sampling to exactly T_out points
        x_ft_traj[f] = constant_speed_2segment(x0_f, x1_best, x2_f, T_out)

    return x_ft_traj


def heuristic_path_initialization(
    x_lft_init,
    x_lft_goal,
    x_rft_init,
    x_rft_goal,
    x_dim,
    num_samples,
    traj_length,
    length,
    constraints_margin,
    dist_func,
):
    """
    Generate collision-heuristic initial trajectories for two fingers (left/right)
    via sampling equally spaced directions on S² or S¹.

    Returns:
        x_lft_traj: (traj_length+2, x_dim) trajectory for left finger
        x_rft_traj: (traj_length+2, x_dim) trajectory for right finger
    """

    x_ft_goal = np.stack([x_lft_goal, x_rft_goal], axis=0)  # (2, x_dim)
    x_ft_init = np.stack([x_lft_init, x_rft_init], axis=0)  # (2, x_dim)

    # check if necessary
    t = np.linspace(0, 1, traj_length + 2).reshape(1, -1, 1)
    x_ft_linear_traj = (1 - t) * x_ft_init[:, None, :] + t * x_ft_goal[
        :, None, :
    ]  # (2, T+2, x_dim)
    dists = dist_func(x_ft_linear_traj.reshape(-1, x_dim))
    dists = dists.reshape(2, traj_length + 2)
    num_violations = (dists < constraints_margin).sum(axis=1)
    if num_violations.sum() == 0:
        # No violations, return linear trajectory
        return x_ft_linear_traj[0], x_ft_linear_traj[1]
    else:
        if x_dim == 2:

            def uniform_unit_vectors_2d(n):
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                return np.stack([np.cos(angles), np.sin(angles)], axis=1)

            u = uniform_unit_vectors_2d(num_samples)

        elif x_dim == 3:

            def fibonacci_sphere(n):
                i = np.arange(n)
                phi = np.arccos(1 - 2 * (i + 0.5) / n)
                theta = np.pi * (1 + 5**0.5) * i
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                return np.stack([x, y, z], axis=1)

            u = fibonacci_sphere(num_samples)

        x_ft_via = x_ft_goal + u[:, None, :] * length  # (num_samples, 2, x_dim)

        # Time interpolation
        t = np.linspace(0, 1, traj_length // 2 + 1).reshape(1, 1, -1, 1)  # (1, 1, T, 1)
        x_ft_init = x_ft_init.reshape(1, 2, 1, x_dim)  # (1, 2, 1, x_dim)
        x_ft_goal = x_ft_goal.reshape(1, 2, 1, x_dim)  # (1, 2, 1, x_dim)
        x_ft_via = x_ft_via[:, :, None, :]  # (num_samples, 2, 1, x_dim)

        # Trajectory candidates: (num_samples, 2, traj_length+2, x_dim)
        ft_traj_candidates = np.concatenate(
            [(1 - t) * x_ft_init + t * x_ft_via, t * x_ft_goal + (1 - t) * x_ft_via],
            axis=2,
        )

        # Evaluate signed distances
        ft_dists = dist_func(ft_traj_candidates.reshape(-1, x_dim))
        ft_dists = ft_dists.reshape(num_samples, 2, -1)

        # Count violations and select best
        num_violations = (ft_dists < constraints_margin).sum(axis=2)
        min_violations = num_violations.min(axis=0)

        best_candidates_lft_idx = np.where(min_violations[0] == num_violations[:, 0])[0]
        best_candidates_rft_idx = np.where(min_violations[1] == num_violations[:, 1])[0]

        # print(f"min_violations: {min_violations}, num_violations: {num_violations}")

        lft_lengths = np.linalg.norm(
            ft_traj_candidates[best_candidates_lft_idx, 0, 1:, :]
            - ft_traj_candidates[best_candidates_lft_idx, 0, :-1, :],
            axis=-1,
        ).sum(axis=-1)

        best_lft_idx = best_candidates_lft_idx[np.argmin(lft_lengths)]

        rft_lengths = np.linalg.norm(
            ft_traj_candidates[best_candidates_rft_idx, 1, 1:, :]
            - ft_traj_candidates[best_candidates_rft_idx, 1, :-1, :],
            axis=-1,
        ).sum(axis=-1)
        best_rft_idx = best_candidates_rft_idx[np.argmin(rft_lengths)]

        x_lft_init_best, x_lft_via_best, x_lft_goal_best = (
            x_ft_init[0, 0, 0],
            x_ft_via[best_lft_idx, 0, 0],
            x_ft_goal[0, 0, 0],
        )
        x_rft_init_best, x_rft_via_best, x_rft_goal_best = (
            x_ft_init[0, 1, 0],
            x_ft_via[best_rft_idx, 1, 0],
            x_ft_goal[0, 1, 0],
        )

        def constant_speed_2segment(x0, x1, x2, n):
            """Return n points sampled uniformly along line x0 → x1 → x2"""
            l1 = np.linalg.norm(x1 - x0)
            l2 = np.linalg.norm(x2 - x1)
            total = l1 + l2
            t = np.linspace(0, 1, n)
            pts = np.where(
                t[:, None] <= l1 / total,
                x0 + (x1 - x0) * (t[:, None] * total / l1),
                x1 + (x2 - x1) * ((t[:, None] - l1 / total) * total / l2),
            )
            return pts

        num_output_points = traj_length + 2
        x_lft_traj = constant_speed_2segment(
            x_lft_init_best, x_lft_via_best, x_lft_goal_best, num_output_points
        )
        x_rft_traj = constant_speed_2segment(
            x_rft_init_best, x_rft_via_best, x_rft_goal_best, num_output_points
        )

        return x_lft_traj, x_rft_traj
