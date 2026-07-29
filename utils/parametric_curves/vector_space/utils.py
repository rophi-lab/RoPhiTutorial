import numpy as np


def get_basis_parameters(basis: str, num_basis: int):
    """
    Returns the parameters for the specified basis function.
    :param basis: Type of basis function (e.g., "Gaussian", "Polynomial", etc.)
    :param num_basis: Number of basis functions
    :return: Dictionary of basis parameters
    """
    if basis == "Gaussian":
        return {"mean": np.linspace(0, 1, num_basis), "std": 1 / num_basis}
    else:
        raise ValueError(f"Unknown basis function: {basis}")


def get_t_traj(
    q_traj: np.ndarray, tf: float, mode: str = "constant_speed_with_smooth_decay"
):
    """
    Returns the time trajectory for the given trajectory.
    :param q_traj: (n, dim) trajectory
    :param tf: terminal time
    :return: (n, 1) time trajectory
    """
    if mode == "equal_interval":
        return np.linspace(0, tf, len(q_traj)).reshape(-1, 1)
    elif mode == "constant_speed_with_smooth_decay":
        q = np.asarray(q_traj, dtype=float)
        assert q.ndim == 2 and q.shape[0] >= 2, "q_traj must be (N,d), N>=2"

        # segment lengths and cumulative arc length
        seg_len = np.linalg.norm(q[1:] - q[:-1], axis=1)  # (N-1,)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # (N,)
        L = cum[-1]

        if L <= 1e-12:  # degenerate: all points identical
            return np.linspace(0.0, tf, len(q)).reshape(-1, 1)

        t_traj = tf * (cum / L)  # proportional split
        return t_traj.reshape(-1, 1)
