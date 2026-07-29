import numpy as np


def skew(w):
    """Returns the skew-symmetric matrix of a 3D vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def unskew(W):
    """Returns the 3D vector from a skew-symmetric matrix."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def exp_so3(w):
    """Returns the matrix exponential of a skew-symmetric matrix."""
    if w.shape == (3, 3):
        w = unskew(w)
    theta = np.linalg.norm(w)
    w_skew = skew(w)
    if theta < 1e-5:
        return np.eye(3) + w_skew + (1 / 2) * (w_skew @ w_skew)
    else:
        return (
            np.eye(3)
            + np.sin(theta) / theta * w_skew
            + (1 - np.cos(theta)) / (theta**2) * (w_skew @ w_skew)
        )


def log_SO3(R):
    """Returns the logarithm of a rotation matrix."""
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)
    if theta < 1e-5:
        return (R - R.T) / 2
    W = (R - R.T) * theta / (2 * np.sin(theta))
    return W


def batch_skew(w):
    """Returns the skew-symmetric matrix of a batch of 3D vectors."""
    W = np.zeros((w.shape[0], 3, 3))
    W[:, 0, 1] = -w[:, 2]
    W[:, 0, 2] = w[:, 1]
    W[:, 1, 0] = w[:, 2]
    W[:, 1, 2] = -w[:, 0]
    W[:, 2, 0] = -w[:, 1]
    W[:, 2, 1] = w[:, 0]
    return W


def batch_unskew(W):
    """Returns the 3D vectors from a batch of skew-symmetric matrices."""
    w = np.zeros((W.shape[0], 3))
    w[:, 0] = W[:, 2, 1]
    w[:, 1] = W[:, 0, 2]
    w[:, 2] = W[:, 1, 0]
    return w
