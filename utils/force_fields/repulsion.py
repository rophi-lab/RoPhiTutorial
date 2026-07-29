import numpy as np


#######################################
# REAL-TIME OBSTACLE AVOIDANCE        #
# FOR M.ANIPULATORS AND MQUIEE ROBOTS #
# Artificial Potential Fields (APF)   #
#######################################
def APF(
    distance: np.ndarray,
    normal: np.ndarray,
    repulsion_factor: float = 0.1,
    threshold: float = 0.01,
    max_value: float = 5.0,
    order1: int = 1,
    order2: int = 1,
):
    """
    distance: (n, )
    normal: (n, dim) unit normal vector pointing away from the obstacle
    """
    normal = normal / np.clip(
        np.linalg.norm(normal, axis=1, keepdims=True), a_min=1e-6, a_max=None
    )

    vel = np.zeros_like(normal)
    repulsive_idx = distance - threshold < 0
    if not np.any(repulsive_idx):
        pass
    else:
        rho = distance[repulsive_idx]
        rho_0 = threshold
        unit_drho_dx = normal[repulsive_idx]
        abs_rho = np.abs(rho.reshape(-1, 1))
        vel[repulsive_idx] = (
            repulsion_factor
            * (1 / abs_rho**order1 - 1 / rho_0**order1)
            * 1
            / abs_rho**order2
            * unit_drho_dx
        )
        if not np.any(rho < 0):
            pass
        else:
            vel[repulsive_idx][rho < 0] = max_value * unit_drho_dx[rho < 0]
    return np.clip(vel, a_min=None, a_max=max_value)


##########################################
# Quadratic Repulsive Force Fields QRFF) #
##########################################
def QRFF(
    distance: np.ndarray,
    normal: np.ndarray,
    repulsion_factor: float = 10000,
    threshold: float = 0.01,
    max_value: float = 5.0,
):
    """
    distance: (n, )
    normal: (n, dim) unit normal vector pointing away from the obstacle
    """
    normal = normal / np.clip(
        np.linalg.norm(normal, axis=1, keepdims=True), a_min=1e-6, a_max=None
    )

    vel = np.zeros_like(normal)
    repulsive_idx = distance - threshold < 0
    if not np.any(repulsive_idx):
        pass
    else:
        rho = distance[repulsive_idx].reshape(-1, 1)
        unit_drho_dx = normal[repulsive_idx]
        vel[repulsive_idx] = repulsion_factor * ((rho - threshold) ** 2) * unit_drho_dx
    # Even when rho is negative, the force is still directed away from the obstacle.
    return np.clip(vel, a_min=None, a_max=max_value)
