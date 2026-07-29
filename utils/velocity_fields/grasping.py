import numpy as np


def projected_gradient_descent(x_lft, x_rft, nl, nr, step_size=0.1):
    """
    Perform projected gradient descent to find the antipodal grasp.
    """
    u = x_rft - x_lft
    u /= np.clip(np.linalg.norm(u), a_min=1e-6, a_max=None)
    nl_dot_u = np.dot(nl, u)
    nr_dot_u = np.dot(nr, u)
    L = 0.5 * (np.arccos(nl_dot_u) + np.arccos(-nr_dot_u))  # mean angle

    du_dxrft = np.eye(3) - np.outer(u, u)
    du_dxlft = -du_dxrft
    dL_du = 0.5 * (
        -1 / np.sqrt(np.clip(1 - nl_dot_u**2, a_min=1e-6, a_max=None)) * nl
        + 1 / np.sqrt(np.clip(1 - nr_dot_u**2, a_min=1e-6, a_max=None)) * nr
    )

    xd_lft_des = -du_dxlft @ dL_du
    xd_rft_des = -du_dxrft @ dL_du

    xd_lft_des -= np.dot(xd_lft_des, nl) * nl
    xd_rft_des -= np.dot(xd_rft_des, nr) * nr

    xd_lft_des *= step_size
    xd_rft_des *= step_size
    return xd_lft_des, xd_rft_des


def cross_finger_gradient_descent(x_lft, x_rft, nl, nr, step_size=0.1):
    """
    Perform cross-finger gradient descent to find the antipodal grasp.
    """
    u = x_rft - x_lft
    u /= np.clip(np.linalg.norm(u), a_min=1e-6, a_max=None)
    nl_dot_u = np.dot(nl, u)
    nr_dot_u = np.dot(nr, u)
    # L = 0.5 * (np.arccos(nl_dot_u) + np.arccos(-nr_dot_u))  # mean angle

    du_dxrft = np.eye(3) - np.outer(u, u)
    du_dxlft = -du_dxrft
    dthetalft_du = 0.5 * (
        -1 / np.sqrt(np.clip(1 - nl_dot_u**2, a_min=1e-12, a_max=None)) * nl
    )
    dthetarft_du = 0.5 * (
        1 / np.sqrt(np.clip(1 - nr_dot_u**2, a_min=1e-12, a_max=None)) * nr
    )

    xd_lft_des = -du_dxlft @ dthetarft_du
    xd_rft_des = -du_dxrft @ dthetalft_du

    xd_lft_des -= np.dot(xd_lft_des, nl) * nl
    xd_rft_des -= np.dot(xd_rft_des, nr) * nr

    xd_lft_des *= step_size
    xd_rft_des *= step_size
    return xd_lft_des, xd_rft_des

def goto_x_des_const_then_linear_clamped_componentwise(x, x_des, v, eps, control_dt):
    """
    Component-wise version of goto_x_des_const_then_linear_clamped.
    """
    diff = x_des - x
    e = np.abs(diff)

    # unit direction
    u = np.sign(diff)

    # speed schedule: constant then linear
    # e >= eps  -> speed = v
    # e <  eps  -> speed = v * (e/eps)
    speed = np.where(e >= eps, v, v * (e / eps))

    # raw velocity
    xdot = speed * u

    # ---- clamping trick (dt-aware): prevent overshoot in one step ----
    max_step = e  # never move more than remaining distance
    step_norm = np.abs(xdot) * control_dt
    overshoot_mask = step_norm > max_step
    xdot[overshoot_mask] = (max_step[overshoot_mask] / control_dt) * u[overshoot_mask]

    return xdot

def goto_x_des_const_then_linear_clamped(x, x_des, v, eps, control_dt):
    """
    Constant speed v when far (e >= eps),
    linear decay to 0 when near (e < eps),
    and clamp so one control step cannot overshoot x_des.

    Args:
        x, x_des: (d,) vectors
        v: max speed (scalar)
        eps: linear region radius (scalar > 0)
        control_dt: controller timestep (scalar > 0)

    Returns:
        xdot: (d,) velocity command
    """
    diff = x_des - x
    e = np.linalg.norm(diff)

    if e < 1e-12:
        return np.zeros_like(x)

    # unit direction
    u = diff / e

    # speed schedule: constant then linear
    # e >= eps  -> speed = v
    # e <  eps  -> speed = v * (e/eps)
    speed = v if e >= eps else v * (e / eps)

    # raw velocity
    xdot = speed * u

    # ---- clamping trick (dt-aware): prevent overshoot in one step ----
    # Proposed step: dx = xdot * dt. If ||dx|| > e, scale down so ||dx|| == e.
    max_step = e  # never move more than remaining distance
    step_norm = speed * control_dt
    if step_norm > max_step:
        # equivalent to setting speed = e/dt
        xdot = (max_step / control_dt) * u

    return xdot


def goto_x_des_with_v_quad_decay_from_eps(x, x_des, v, eps, arm_only=False):
    diff = x_des - x
    if arm_only:
        e = np.linalg.norm(diff[:7])
    else:
        e = np.linalg.norm(diff)
    if e > eps:
        return v * diff / e
    elif e < 1.0e-6:
        return np.zeros_like(x)
    else:
        return v * (1 - (e - eps) ** 2 / (eps) ** 2) * diff / e


def goto_x_hat_des_with_w_quad_decay_from_eps(x_hat, x_hat_des, w, eps):
    e = np.arccos(np.clip((x_hat_des * x_hat).sum(), a_min=-1, a_max=1))
    _dir = np.cross(x_hat, x_hat_des)
    if e > eps:
        return w * _dir
    elif e < 1.0e-6:
        return np.zeros_like(x_hat)
    else:
        return w * (1 - (e - eps) ** 2 / (eps) ** 2) * _dir


def two_finger_reaching_vf(
    x_cetner,  # (3,)
    x_diff,  # (3,)
    x_center_des,  # (3,)
    x_diff_des,  # (3,)
    nhat,  # (3,)
    phi_T=2 * np.pi,
    err_T=0.02,
    err_thr=0.01,
    Vphi=1,
    Vc=10,
    Vd=100,
    Ad_max=1.5,
    Ad_min=0.5,
):
    # compute finger tip center reaching error
    x_center_disp = x_center_des - x_cetner
    error = np.linalg.norm(x_center_disp)
    ehat = x_center_disp / np.clip(error, a_min=1.0e-6, a_max=np.inf)

    # velocity of the finger tip difference (depends on the finger tip center error)
    weight_err = np.tanh((error - err_thr) / err_T)
    xd_dot_des = (1 + weight_err) / 2 * (Ad_max * x_diff_des - x_diff) + (
        1 - weight_err
    ) / 2 * (Ad_min * x_diff_des - x_diff)

    # velocity of the finger tip center (weighted sum of reaching and aligning ehat and nhat)
    phi = np.arccos(
        np.clip((ehat * nhat).sum(), a_min=-1, a_max=1)
    )  # -pi ~ pi (pi: aligned, 0: perpendicular)
    weight_phi = np.exp(-(np.pi - phi) / phi_T)
    phi_err = np.pi - phi
    xc_dot_des = weight_phi * x_center_disp + Vphi * phi_err * (
        nhat - ehat * (ehat * nhat).sum()
    ) * (1 - weight_phi)

    # coordinate transformation
    x_lft_dot_des = Vc * xc_dot_des + Vd * xd_dot_des
    x_rft_dot_des = Vc * xc_dot_des - Vd * xd_dot_des
    return np.hstack((x_lft_dot_des, x_rft_dot_des))  # (6,)


def gripper_pose_vf(
    x_cetner, x_center_des, q_gripper, q_gripper_nominal, err_thr=0.1, err_T=0.01, V=1
):
    error = np.linalg.norm((x_center_des - x_cetner))
    weight = np.exp(-np.clip(error - err_thr, a_min=0, a_max=np.inf) / err_T)
    v1 = q_gripper_nominal - q_gripper

    v2 = np.zeros_like(q_gripper)
    v2[0] = q_gripper[4] - q_gripper[0]
    v2[4] = q_gripper[0] - q_gripper[4]
    q_dot_des = (1 - weight) * v1 + weight * v2
    return V * q_dot_des


def align_wrist_vf(p, xc_des, nhat, V=1):
    # align the base position onto the nhat
    e = xc_des - p
    error = np.linalg.norm(e)
    ehat = e / np.clip(error, a_min=1.0e-6, a_max=np.inf)
    phi = np.arccos(np.clip((ehat * nhat).sum(), a_min=-1, a_max=1))
    phi_err = np.pi - phi
    p_dot_des = phi_err * (nhat - ehat * (ehat * nhat).sum())
    return V * p_dot_des


def _so3_logvee(R_err: np.ndarray) -> np.ndarray:
    """
    Log map from SO(3) to R^3: phi = log(R_err)^vee.
    Numerically stable near 0.
    """
    tr = np.trace(R_err)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # small-angle approximation
    if theta < 1e-12:
        w_hat = 0.5 * (R_err - R_err.T)
        return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64)

    w_hat = (theta / (2.0 * np.sin(theta))) * (R_err - R_err.T)
    return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64)


def goto_R_des_const_then_linear_clamped_world(R, R_des, omega, eps, control_dt):
    """
    World/spatial-frame version.

    Constant angular speed omega when far (theta >= eps),
    linear decay to 0 when near (theta < eps),
    and clamp so one control step cannot overshoot R_des.

    Args:
        R, R_des: (3,3) rotation matrices (SO(3))
        omega: max angular speed [rad/s] (scalar)
        eps: linear region radius [rad] (scalar > 0)
        control_dt: controller timestep [s] (scalar > 0)

    Returns:
        omega_world: (3,) angular velocity command in WORLD/SPATIAL frame.
    """
    # Spatial/world-frame error: rotates current -> desired in world coordinates
    R_err = R_des @ R.T

    # World-frame axis-angle vector: phi = theta * u (in world frame)
    phi = _so3_logvee(R_err)
    theta = np.linalg.norm(phi)

    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    u = phi / theta

    # speed schedule (rad/s)
    speed = omega if theta >= eps else omega * (theta / eps)

    # raw world-frame angular velocity
    omega_world = speed * u

    # dt-aware clamp: ||omega_world|| * dt <= theta
    step_angle = speed * control_dt
    if step_angle > theta:
        omega_world = (theta / control_dt) * u

    return omega_world
