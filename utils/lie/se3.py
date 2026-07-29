import numpy as np

from utils.lie.so3 import skew, unskew, batch_skew, batch_unskew, log_SO3


def vec_to_se3(xi):
    """
    Converts a se(3) vector to its corresponding se(3) hat matrix.

    Parameters
    ----------
    xi : np.ndarray
        A 6x1 vector representing the se(3) vector.

    Returns
    -------
    np.ndarray
        A 4x4 matrix representing the se(3) hat matrix.
    """
    assert xi.shape == (6,), "Input must be a 6x1 vector."

    omega = xi[:3]
    v = xi[3:]

    omega_hat = skew(omega)

    se3_hat = np.zeros((4, 4))
    se3_hat[:3, :3] = omega_hat
    se3_hat[:3, 3] = v
    return se3_hat


def batch_vec_to_se3(xi):
    """
    Converts a batch of se(3) vectors to their corresponding se(3) hat matrices.

    Parameters
    ----------
    xi : np.ndarray
        A (N, 6) matrix representing N se(3) vectors.

    Returns
    -------
    np.ndarray
        A (N, 4, 4) matrix representing the se(3) hat matrices.
    """
    assert xi.shape[1] == 6, "Input must be a (N, 6) matrix."

    omega = xi[:, :3]
    v = xi[:, 3:]

    omega_hat = batch_skew(omega)

    se3_hat = np.zeros((xi.shape[0], 4, 4))
    se3_hat[:, :3, :3] = omega_hat
    se3_hat[:, :3, 3] = v
    return se3_hat


def se3_to_vec(xi_hat):
    """
    Converts a se(3) hat matrix to its corresponding se(3) vector.

    Parameters
    ----------
    xi_hat : np.ndarray
        A 4x4 matrix representing the se(3) hat matrix.

    Returns
    -------
    np.ndarray
        A 6x1 vector representing the se(3) vector.
    """
    assert xi_hat.shape == (4, 4), "Input must be a 4x4 matrix."

    omega_hat = xi_hat[:3, :3]
    v = xi_hat[:3, 3]

    omega = unskew(omega_hat)

    return np.hstack((omega, v))


def batch_se3_to_vec(xi_hat):
    """
    Converts a batch of se(3) hat matrices to their corresponding se(3) vectors.

    Parameters
    ----------
    xi_hat : np.ndarray
        A (N, 4, 4) matrix representing N se(3) hat matrices.

    Returns
    -------
    np.ndarray
        A (N, 6) matrix representing the se(3) vectors.
    """
    assert xi_hat.shape[1:] == (4, 4), "Input must be a (N, 4, 4) matrix."

    omega_hat = xi_hat[:, :3, :3]
    v = xi_hat[:, :3, 3]

    omega = batch_unskew(omega_hat)

    return np.hstack((omega, v))


def invSE3(T):
    """
    Computes the inverse of a SE(3) transformation matrix.

    Parameters
    ----------
    T : np.ndarray
        A 4x4 matrix representing the SE(3) transformation.

    Returns
    -------
    np.ndarray
        A 4x4 matrix representing the inverse SE(3) transformation.
    """
    assert T.shape == (4, 4), "Input must be a 4x4 matrix."

    R = T[:3, :3]
    p = T[:3, 3]

    R_inv = R.T
    p_inv = -R_inv @ p

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = p_inv
    return T_inv


def batch_invSE3(T):
    """
    Computes the inverse of a batch of SE(3) transformation matrices.

    Parameters
    ----------
    T : np.ndarray
        A (N, 4, 4) matrix representing N SE(3) transformations.

    Returns
    -------
    np.ndarray
        A (N, 4, 4) matrix representing the inverse SE(3) transformations.
    """
    assert T.shape[1:] == (4, 4), "Input must be a (N, 4, 4) matrix."

    R = T[:, :3, :3]
    p = T[:, :3, 3]

    R_inv = np.transpose(R, axes=(0, 2, 1))
    p_inv = -np.einsum("ijk,ik->ij", R_inv, p)

    T_inv = np.zeros((T.shape[0], 4, 4))
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = p_inv
    T_inv[:, 3, 3] = 1
    return T_inv


def log_SE3(T):
    """
    Computes the logarithm of a SE(3) transformation matrix.

    Parameters
    ----------
    T : np.ndarray
        A 4x4 matrix representing the SE(3) transformation.

    Returns
    -------
    np.ndarray
        A 4x4 matrix representing the se(3) vector.
    """
    angle_threshold = 1e-6
    R = T[:3, :3]
    trace = np.trace(R)
    skew_S = np.zeros((4, 4))
    if np.abs(trace - 3) < angle_threshold:
        skew_S[:3, 3] = T[:3, 3]
    if np.abs(trace - 3) >= angle_threshold:
        skew_w = log_SO3(R)
        theta = np.arccos(
            np.clip(
                0.5 * (trace - 1), a_min=-1 + 1e-6, a_max=1 - 1e-6
            )  # Avoid numerical issues
        )
        wmat = skew_w / theta
        identity = np.eye(3)
        invG = (
            (1 / theta) * identity
            - 0.5 * wmat
            + (1 / theta - 0.5 / np.tan(0.5 * theta)) * wmat @ wmat
        )
        skew_S[:3, :3] = skew_w
        skew_S[:3, 3] = (theta * (invG @ T[:3, 3:4])).reshape(3)
    return skew_S


def exp_se3(x):
    # assert v.ndim == 1
    TOLERANCE = 1e-6
    X = np.eye(4)
    R = None
    Jl = None
    w = unskew(x[:3, :3])
    theta = np.linalg.norm(w)
    I = np.eye(3)
    v = x[:3, 3]

    if theta < TOLERANCE:
        R = I
        Jl = I
    else:
        A = skew(w)
        theta2 = theta * theta
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        oneMinusCosTheta2 = (1 - ctheta) / (theta2)
        A2 = A @ A
        R = I + (stheta / theta) * A + oneMinusCosTheta2 * A2
        Jl = I + oneMinusCosTheta2 * A + ((theta - stheta) / (theta2 * theta)) * A2

    X[0:3, 0:3] = R
    X[0:3, 3] = (Jl @ v).reshape(3)

    return X


def AdjointSE3(T):
    """
    Computes the adjoint representation of a SE(3) transformation matrix.

    Parameters
    ----------
    T : np.ndarray
        A 4x4 matrix representing the SE(3) transformation.

    Returns
    -------
    np.ndarray
        A 6x6 matrix representing the adjoint representation.
    """
    assert T.shape == (4, 4), "Input must be a 4x4 matrix."

    R = T[:3, :3]
    p = T[:3, 3]

    p_hat = skew(p)

    # Construct the adjoint matrix
    # [[R, 0]
    # [p_hat * R, R]]
    Adjoint = np.zeros((6, 6))
    Adjoint[:3, :3] = R
    Adjoint[3:, :3] = p_hat @ R
    Adjoint[:3, 3:] = np.zeros((3, 3))
    Adjoint[3:, 3:] = R

    return Adjoint
