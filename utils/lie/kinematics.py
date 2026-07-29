import numpy as np
from utils.lie.so3 import skew


def get_point_Jacobian(
    R: np.ndarray,
    J_p: np.ndarray,
    J_R: np.ndarray,
    Delta: np.ndarray,
):
    """
    Compute the Jacobian of a point attached to a rigid body.
    The right body pose is (p, R) and its Jacobian is (J_p, J_R).
    Delta is the point expreessed in the body frame.

    Let x be the pose of the point in the world frame.
    x = p + R Delta
    J_x qd = J_p qd + [J_R qd] R Delta
    J_x = J_p - [R Delta] J_R
    """
    J_x = J_p - skew(R @ Delta) @ J_R
    return J_x
