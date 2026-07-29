import numpy as np

from utils.lie.so3 import exp_so3, log_SO3, skew, unskew
from utils.lie.se3 import (
    batch_invSE3,
    batch_se3_to_vec,
    invSE3,
    se3_to_vec,
    logSE3,
    AdjointSE3,
)


def plan_Rn_constant_speed_trajectory(
    q_start: np.ndarray, q_end: np.ndarray, T: float, num_points: int
):
    """
    Plan a trajectory in joint space with constant speed.

    Args:
        q_start (np.ndarray): Starting joint configuration.
        q_end (np.ndarray): Ending joint configuration.
        T (float): Total time for the trajectory.
        num_points (int): Number of points in the trajectory.

    Returns:
        np.ndarray: Time trajectory.
        np.ndarray: Planned trajectory.
    """
    # Calculate the time trajectory
    t_traj = np.linspace(0, T, num_points)

    # Calculate the trajectory in joint space
    q_traj = np.array([q_start + (q_end - q_start) * (t / T) for t in t_traj])
    return t_traj, q_traj


def plan_SE3_constant_speed_trajectory(
    T_start: np.ndarray, T_end: np.ndarray, T: float, num_points: int
):
    """
    Plan a trajectory in SE(3) with constant speed.
    Args:
        T_start (np.ndarray): Starting transformation matrix.
        T_end (np.ndarray): Ending transformation matrix.
        T (float): Total time for the trajectory.
        num_points (int): Number of points in the trajectory.
    Returns:
        np.ndarray: Time trajectory.
        np.ndarray: Planned trajectory.
    """
    # Calculate the time trajectory
    t_traj = np.linspace(0, T, num_points)
    T_traj = np.zeros((num_points, 4, 4))

    R_start = T_start[:3, :3]
    p_start = T_start[:3, 3]
    R_end = T_end[:3, :3]
    p_end = T_end[:3, 3]

    for i in range(num_points):
        t = i * T / (num_points - 1)
        T_traj[i, :3, :3] = R_start @ exp_so3(log_SO3(R_start.T @ R_end) * (t / T))
        T_traj[i, :3, 3] = (1 - t / T) * p_start + t / T * p_end
        T_traj[i, 3, 3] = 1.0
    return t_traj, T_traj
