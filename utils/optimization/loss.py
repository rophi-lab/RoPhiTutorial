import numpy as np

from utils.lie.se3 import invSE3


def HuberLoss(x, y, delta):
    residual = x - y
    abs_residual = np.linalg.norm(residual)
    if abs_residual > delta:
        dloss_dx = delta * residual / np.clip(abs_residual, a_min=1e-10, a_max=None)
        loss = delta * (abs_residual - 0.5 * delta)
    else:
        dloss_dx = residual
        loss = 0.5 * abs_residual**2
    return loss, dloss_dx


def L2Loss(x, y):
    residual = x - y
    loss = 0.5 * np.sum(residual**2)
    dloss_dx = residual
    return loss, dloss_dx


def HuberLoss3DPointTo2DCone(
    x: np.ndarray,
    n: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    theta: float,
    delta: float,
):
    """
    Huber loss for 3D point to 2D cone distance.
    Args:
        x: 3D point (3,)
        n: 2d plane normal (3,)
        r: cone axis (3,)
        c: cone center (3,)
        theta: cone half-angle (float)
        delta: Huber loss parameter (float)
    """
    n = n / np.linalg.norm(n)
    r = r / np.linalg.norm(r)
    e = np.cross(r, n)

    # project x to the plane
    inner_prod = np.sum((x - c) * n)
    x_to_plane = inner_prod * n  # (3,)
    x_to_plane_dist = np.linalg.norm(x_to_plane)  # (1,)
    dx_to_plane_dist_dx = np.sign(inner_prod) * n

    x_proj = x - x_to_plane  # (3,)

    # transformation
    T = np.concatenate(
        [e.reshape(3, 1), r.reshape(3, 1), n.reshape(3, 1), c.reshape(3, 1)], axis=1
    )  # (3, 4)
    T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)  # (4, 4)
    T_inv = invSE3(T)  # (4, 4)
    R_inv = T_inv[:3, :3]  # (3, 3)
    t_inv = T_inv[:3, 3:]  # (3, 1)

    x_proj_tf = (R_inv @ x_proj.reshape(3, 1) + t_inv).flatten()  # (3,)

    # angle from projected x axis (0, 2pi)
    a = x_proj_tf[0]  # x
    b = x_proj_tf[1]  # y
    angle = np.arctan2(b, a)
    angle = (angle + 2 * np.pi) % (2 * np.pi)

    # Projection matrix to the plane: I - n n^T
    P_plane = np.eye(3) - np.outer(n, n)

    # da/dx = R_inv[0, :] @ P_plane
    da_dx = R_inv[0, :] @ P_plane  # (3,)
    db_dx = R_inv[1, :] @ P_plane  # (3,)

    if (angle >= np.pi / 2 - theta) and (angle < np.pi / 2 + theta):  # inside cone
        distance_within_plane = 0.0
        ddistance_within_plane_da = np.zeros(3)
        ddistance_within_plane_db = np.zeros(3)
    elif (angle >= np.pi / 2 + theta) and (angle < np.pi + theta):  # left side cone
        k = np.tan(np.pi / 2 - theta)
        denom = np.sqrt(k**2 + 1)
        distance_within_plane = np.abs(k * a - b) / denom
        sign_term = np.sign(k * a - b)
        ddistance_within_plane_da = (sign_term * k) / denom
        ddistance_within_plane_db = (-sign_term) / denom
    elif (angle >= np.pi + theta) and (
        angle < 2 * np.pi - theta
    ):  # bottom side cone (nearest point is the origin)
        distance_within_plane = np.sqrt(a**2 + b**2)
        denom = np.clip(
            distance_within_plane, a_min=1e-6, a_max=None
        )  # add epsilon to avoid division by zero
        ddistance_within_plane_da = a / denom
        ddistance_within_plane_db = b / denom
    else:  # right side cone
        k = np.tan(np.pi / 2 + theta)
        denom = np.sqrt(k**2 + 1)
        distance_within_plane = np.abs(k * a - b) / denom
        sign_term = np.sign(k * a - b)
        ddistance_within_plane_da = (sign_term * k) / denom
        ddistance_within_plane_db = (-sign_term) / denom
    ddistance_within_plane_dx = (
        ddistance_within_plane_da * da_dx + ddistance_within_plane_db * db_dx
    )

    dist = np.sqrt(distance_within_plane**2 + x_to_plane_dist**2)
    ddist_dx = (
        distance_within_plane * ddistance_within_plane_dx
        + x_to_plane_dist * dx_to_plane_dist_dx
    ) / np.clip(
        dist, a_min=1e-10, a_max=None
    )  # add epsilon to avoid division by zero

    if dist > delta:
        loss = delta * (dist - 0.5 * delta)
        dloss_dx = delta * ddist_dx
    else:
        loss = 0.5 * dist**2
        dloss_dx = dist * ddist_dx
    return dist, ddist_dx, loss, dloss_dx


def HuberLossTwoPointsPointingOnePoint(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    delta: float,
):
    """
    Huber loss for two points pointing to one point.
    Args:
        x1: 3D point 1 (3,)
        x2: 3D point 2 (3,)
        x3: 3D point 3 (3,)
        delta: Huber loss parameter (float)
    """
    v = x2 - x1  # (3,)
    u = x3 - x1  # (3,)
    vTv = np.dot(v, v)
    vTu = np.dot(v, u)

    t = vTu / vTv
    p = x1 + t * v
    r = p - x3
    d = np.linalg.norm(r)

    # unit direction from x3 to p
    if d < 1e-10:
        r_hat = np.zeros(3)
    else:
        r_hat = r / d

    if d > delta:
        loss = delta * (d - 0.5 * delta)
        dL_dd = delta
    else:
        loss = 0.5 * d**2
        dL_dd = d

    # ∂t/∂x1 and ∂t/∂x2
    dv_dx1 = -np.eye(3)
    dv_dx2 = np.eye(3)
    du_dx1 = -np.eye(3)

    dt_dx1 = (dv_dx1 @ u + v @ du_dx1.T - 2 * vTu / vTv * dv_dx1 @ v) / vTv
    dt_dx2 = (dv_dx2 @ u - 2 * vTu / vTv * dv_dx2 @ v) / vTv

    dp_dx1 = np.eye(3) + t * dv_dx1 + np.outer(v, dt_dx1)
    dp_dx2 = t * dv_dx2 + np.outer(v, dt_dx2)

    dloss_dx1 = dL_dd * dp_dx1.T @ r_hat
    dloss_dx2 = dL_dd * dp_dx2.T @ r_hat

    return loss, dloss_dx1, dloss_dx2


def HuberPalmAlignAngleError(
    x: np.ndarray,
    n: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    theta: float,
    delta: float,
):
    """
    Huber loss for 3D point to 2D cone distance.
    Args:
        x: 3D point (3,)
        n: 2d plane normal (3,)
        r: cone axis (3,)
        c: cone center (3,)
        theta: cone half-angle (float)
        delta: Huber loss parameter (float)
    """
    n = n / np.linalg.norm(n)
    r = r / np.linalg.norm(r)
    e = np.cross(r, n)

    # Project x to the plane
    inner_prod = np.dot(x - c, n)
    x_proj = x - inner_prod * n

    # Compute unit vector from c to x
    vec1 = x - c
    vec1_norm = np.linalg.norm(vec1)
    u = vec1 / vec1_norm

    vec2 = x_proj - c
    vec2_norm = np.linalg.norm(vec2)
    u_prime = vec2 / vec2_norm

    # Compute angle between vec1 and plane normal n
    s = np.clip(np.dot(u, u_prime), -1.0, 1.0)
    Angle1 = np.arccos(s)

    # Compute derivative of Angle1 w.r.t. x
    dAngle1_ds = -1.0 / np.sqrt(1.0 - s**2)

    # du/dx
    I = np.eye(3)
    du_dx = (I - np.outer(u, u)) / vec1_norm

    # dx_proj/dx
    dxproj_dx = I - np.outer(n, n)

    # du'/dx
    du_prime_dx = (I - np.outer(u_prime, u_prime)) / vec2_norm @ dxproj_dx

    # ds/dx
    ds_dx = du_dx.T @ u_prime + u.T @ du_prime_dx

    # Chain rule
    dAngle1_dx = dAngle1_ds * ds_dx  # (3,)

    # transformation
    T = np.concatenate(
        [e.reshape(3, 1), r.reshape(3, 1), n.reshape(3, 1), c.reshape(3, 1)], axis=1
    )  # (3, 4)
    T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)  # (4, 4)
    T_inv = invSE3(T)  # (4, 4)
    R_inv = T_inv[:3, :3]  # (3, 3)
    t_inv = T_inv[:3, 3:]  # (3, 1)

    x_proj_tf = (R_inv @ x_proj.reshape(3, 1) + t_inv).flatten()  # (3,)

    # angle from projected x axis
    a = x_proj_tf[0]  # x
    b = x_proj_tf[1]  # y
    angle = np.arctan2(b, a) + np.pi / 2  # (-1/2 pi, 3/2 pi)

    # Projection matrix to the plane: I - n n^T
    P_plane = np.eye(3) - np.outer(n, n)

    # da/dx = R_inv[0, :] @ P_plane
    da_dx = R_inv[0, :] @ P_plane  # (3,)
    db_dx = R_inv[1, :] @ P_plane  # (3,)
    denom = a**2 + b**2 + 1e-12
    dangle_dx = (-b / denom) * da_dx + (a / denom) * db_dx

    if (angle >= -np.pi / 2) and (angle < np.pi / 2 - theta):
        Angle2 = np.pi / 2 - theta - angle
        dAngle2_dx = -dangle_dx
    elif (angle >= np.pi / 2 - theta) and (angle < np.pi / 2 + theta):
        Angle2 = 0.0
        dAngle2_dx = np.zeros(3)
    elif (angle >= np.pi / 2 + theta) and (angle < 3 * np.pi / 2):
        Angle2 = angle - np.pi / 2 - theta
        dAngle2_dx = dangle_dx

    dist = np.sqrt(Angle1**2 + Angle2**2)
    ddist_dx = 1 / dist * (Angle1 * dAngle1_dx + Angle2 * dAngle2_dx)

    if dist > delta:
        loss = delta * (dist - 0.5 * delta)
        dloss_dx = delta * ddist_dx
    else:
        loss = 0.5 * dist**2
        dloss_dx = dist * ddist_dx
    return loss, dloss_dx


def BatchHuberLoss3DPointTo2DCone(
    x: np.ndarray,
    n: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    theta: float,
    delta: float,
):
    """
    Huber loss for 3D point to 2D cone distance.
    Args:
        x: 3D point (n, 3)
        n: 2d plane normal (3,)
        r: cone axis (3,)
        c: cone center (3,)
        theta: cone half-angle (float)
        delta: Huber loss parameter (float)
    """
    n = n / np.linalg.norm(n)
    r = r / np.linalg.norm(r)
    e = np.cross(r, n)

    n = n.reshape(1, 3)
    r = r.reshape(1, 3)
    e = e.reshape(1, 3)
    c = c.reshape(1, 3)

    # project x to the plane
    x_to_plane = np.sum((x - c) * n, axis=1, keepdims=True) * n  # (n, 3)
    x_to_plane_dist = np.linalg.norm(x_to_plane, axis=1, keepdims=True)  # (n, 1)
    x_proj = x - x_to_plane  # (n, 3)

    # transformation
    T = np.concatenate([e.T, r.T, n.T, c.T], axis=1)  # (3, 4)
    T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)  # (4, 4)
    T_inv = invSE3(T)  # (4, 4)
    R_inv = T_inv[:3, :3]  # (3, 3)
    t_inv = T_inv[:3, 3:]  # (3, 1)

    x_proj_tf = (R_inv @ x_proj.T + t_inv).T  # (n, 3)

    # angle from projected x axis (0, 2pi)
    a = x_proj_tf[:, 0]
    b = x_proj_tf[:, 1]
    angle = np.arctan2(b, a) + np.pi  # (n,)

    dist_within_plane = np.zeros_like(angle)  # (n,)
    idx1 = (angle >= np.pi / 2 - theta) and (angle < np.pi / 2 + theta)  # inside cone
    idx2 = (angle >= np.pi / 2 + theta) and (angle < np.pi + theta)  # left side cone
    idx3 = (angle >= np.pi + theta) and (
        angle < 2 * np.pi - theta
    )  # bottom side cone (nearest point is the origin)
    idx4 = not (idx1 or idx2 or idx3)  # right side cone

    dist_within_plane[idx3] = np.sqrt(a[idx3] ** 2 + b[idx3] ** 2)
    left_line_coeff = np.tan(np.pi / 2 - theta)
    dist_within_plane[idx2] = np.abs(left_line_coeff * a[idx2] - b[idx2]) / np.sqrt(
        left_line_coeff**2 + 1
    )
    right_line_coeff = np.tan(np.pi / 2 + theta)
    dist_within_plane[idx4] = np.abs(right_line_coeff * a[idx4] - b[idx4]) / np.sqrt(
        right_line_coeff**2 + 1
    )
    # TODO: Continue this implementation
    raise NotImplementedError("HuberLoss3DPointTo2DCone is not fully implemented yet.")
