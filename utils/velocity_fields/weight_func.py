import numpy as np


def weight_func_tanh(error, error_threshold, error_temperature):
    """
    if error > error_threshold; it converts to 1
    if error < error_threshold; it converts to 0
    """
    z = (error - error_threshold) / error_temperature
    return 0.5 * (1 + np.tanh(z))


def speed_const_then_linear_clamped(error, max_v, eps, control_dt):
    """
    Scalar speed profile with:
      - constant speed max_v when far (error >= eps)
      - linear decay to 0 when near (error < eps): max_v * (error/eps)
      - dt-aware clamp so one step cannot overshoot: speed <= error/control_dt

    Args:
        error: nonnegative scalar distance-to-goal
        max_v: max speed (>= 0)
        eps: radius for linear region (> 0)
        control_dt: controller timestep (> 0)

    Returns:
        speed: nonnegative scalar
    """
    e = float(error)
    if not np.isfinite(e) or e <= 0.0:
        return 0.0

    max_v = float(max_v)
    eps = float(eps)
    control_dt = float(control_dt)

    # constant then linear
    if e >= eps:
        speed = max_v
    else:
        speed = max_v * (e / eps)

    # dt-aware clamp: ensure speed*dt <= error
    speed = min(speed, e / control_dt)

    # numerical safety
    return max(0.0, speed)


def speed_const_then_linear_clamped_vec(error, max_v, eps, control_dt):
    """
    Vectorized speed profile.

    Args:
        error: array-like, nonnegative distances-to-goal
        max_v: scalar max speed (>= 0)
        eps: scalar radius for linear region (> 0)
        control_dt: scalar controller timestep (> 0)

    Returns:
        speed: ndarray, same shape as error
    """
    error = np.asarray(error, dtype=float)

    max_v = float(max_v)
    eps = float(eps)
    control_dt = float(control_dt)

    # initialize output
    speed = np.zeros_like(error)

    # valid entries: finite and positive
    valid = np.isfinite(error) & (error > 0.0)

    e = error[valid]

    # constant then linear profile
    speed_profile = np.where(e >= eps, max_v, max_v * (e / eps))

    # dt-aware clamp: speed * dt <= error
    speed_clamped = np.minimum(speed_profile, e / control_dt)

    # numerical safety
    speed[valid] = np.maximum(0.0, speed_clamped)

    return speed
