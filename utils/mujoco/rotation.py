import numpy as np


def quat2rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def rotmat2quat(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Apply a quaternion rotation to a vector.

    Args:
        quat: Quaternion in (w, x, y, z), shape (4,).
        vec:  Vector in (x, y, z), shape (3,).

    Returns:
        Rotated vector (3,).

    Implements the same formula as the torch version:
        t = 2 * (xyz x v)
        v' = v + w * t + xyz x t
    """
    # Extract vector part of quaternion (x, y, z) and scalar part w
    xyz = quat[1:]          # (3,)
    w = quat[0:1]           # (1,)

    # t = 2 * (xyz x v)
    t = 2.0 * np.cross(xyz, vec, axis=0)  # (3,)

    # v' = v + w * t + xyz x t
    rotated = vec + w * t + np.cross(xyz, t, axis=0)  # (3,)

    return rotated

def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Apply an inverse quaternion rotation to a vector.

        v' = v - w * t + x.cross(t),  where t = 2 * x.cross(v)

    Args:
        quat: Quaternion in (w, x, y, z), shape (4,).
        vec:  Vector in (x, y, z), shape (3,).

    Returns:
        Rotated vector with the same shape as vec: (3,).
    """
    # Extract vector part of quaternion (x, y, z) and scalar part w
    xyz = quat[1:]          # (3,)
    w = quat[0:1]           # (1,)

    # t = 2 * (xyz x v)
    t = 2.0 * np.cross(xyz, vec, axis=0)  # (3,)

    # v' = v - w * t + xyz x t
    rotated = vec - w * t + np.cross(xyz, t, axis=0)  # (3,)

    return rotated

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate for a single quaternion in (w, x, y, z).
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product q = q1 * q2 for single quaternions in (w, x, y, z).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float64)

def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """
    Convert a unit quaternion (w, x, y, z) to an axis-angle vector (3,).

    The vector's magnitude is the rotation angle (rad),
    and its direction is the rotation axis.
    """
    quat = quat * np.sign(quat[0])

    w = quat[0]
    v = quat[1:]  # (x, y, z)
    mag = np.linalg.norm(v)

    half_angle = np.arctan2(mag, w)
    angle = 2.0 * half_angle

    if np.abs(angle) > eps:
        sin_half_angle_over_angle = np.sin(half_angle) / angle
    else:
        # Taylor approximation for small angles
        sin_half_angle_over_angle = 0.5 - angle * angle / 48.0

    return v / sin_half_angle_over_angle

def quat_box_minus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion box-minus operator for single quaternions:
        q_box = q1 ⊖ q2 in R^3

    Defined as:
        qd = q1 * q2^{-1}
        q_box = log(qd)  (axis-angle of relative rotation)

    Args:
        q1: First quaternion (w, x, y, z), shape (4,).
        q2: Second quaternion (w, x, y, z), shape (4,).

    Returns:
        Axis-angle vector (3,) representing rotation from q2 to q1.
    """
    qd = quat_mul(q1, quat_conjugate(q2))  # q1 * q2^{-1}
    return axis_angle_from_quat(qd)