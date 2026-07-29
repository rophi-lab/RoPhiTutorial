import numpy as np


########################################################
# A Dynamical System Approach to                       #
# Realtime Obstacle Avoidance (Autonomous Robots 2012) #
# S.M. Khansari-Zadeh · Aude Billard                   #
# Dynamiocal System Modulation (DSM)                   #
########################################################
def DSM(
    f: np.ndarray,
    distance: np.ndarray,
    distance_grad: np.ndarray,
    rho: float = 1.0,
    repulsion=True,
):
    """
    f: velocity vectors of (n, dim)
    d: distance (or approximate distance) to the obstacle (n, 1) -- zero on the surface
    distance_grad: direction vector that increases the distance to the obstacle (n, dim)
    rho: reactivity factor, default is 1.0 (the higher, the smoother)
    """

    # convert r to unit vector
    r = distance_grad / np.clip(
        np.linalg.norm(distance_grad, axis=1, keepdims=True), a_max=None, a_min=1.0e-6
    )  # (n, dim)

    dim = f.shape[1]
    weight = 1 / (distance + 1) ** (1 / rho)
    lambda_r = (1 - weight).reshape(-1, 1, 1)  # when Gamma = 0, this goes to 0
    lambda_e = (1 + weight).reshape(-1, 1, 1)  # when Gamma = 0, this goes to 2

    proj_matrix = r.reshape(-1, dim, 1) @ r.reshape(-1, 1, dim)  # (n, 3, 3)

    M = lambda_r * proj_matrix + lambda_e * (
        np.eye(dim).reshape(1, dim, dim) - proj_matrix
    )  # (n, 3, 3)
    f_modulated = (M @ f.reshape(-1, dim, 1)).reshape(-1, dim)

    # This is added for additional repulsive behavior.
    if repulsion:
        repulsion_factor = 100
        neg_dist_idx = distance < 0
        repulsive_force = (
            repulsion_factor
            * (distance[neg_dist_idx] ** 2).reshape(-1, 1)
            * r[neg_dist_idx]
        )
        f_modulated[neg_dist_idx] += repulsive_force
    # Observation:
    # When f is aligned with r and Gamma is small,
    # the norm of modualted f is close to 0, which leads to
    # a non-converging behavior.
    return f_modulated


############################################################
# Avoiding Dense and Dynamic Obstacles in Enclosed Spaces: #
# Application to Moving in Crowds (T-RO 2022)              #
# Lukas Huber, Jean-Jacques Slotine, and Aude Billard      #
# DSM with Reference Point (DSMR)                          #
############################################################
def DSMR(
    f: np.ndarray,
    x: np.ndarray,
    x_ref: np.ndarray,
    x_bd: np.ndarray,
    n_bd: np.ndarray,
    rho: float = 1.0,
    p: float = 1.0,
    surface_friction=True,
    repulsion=True,
    c_rep=None,
    output_Gamma_o=False,
):
    """
    f: velocity vectors of (n, dim)
    d: distance (or approximate distance) to the obstacle (n, 1) -- zero on the surface
    x: position of the agent (n, dim)
    x_ref: reference point (dim)
    x_bd: position of the boundary (n, dim)
    n_bd: normal vector of the boundary (n, dim)
    rho: reactivity factor, default is 1.0 (the higher, the smoother)
    repulsion: whether to apply repulsive force, default is True
    """
    dim = f.shape[1]
    r = x - x_ref.reshape(1, dim)  # (n, dim)
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    unit_r = r / np.clip(r_norm, a_min=1.0e-6, a_max=None)  # (n, dim)
    R = np.linalg.norm(x_bd - x_ref.reshape(1, dim), axis=1, keepdims=True)
    Gamma_o = (r_norm / R) ** (2 * p)

    e1 = n_bd + np.random.randn(1, dim)
    e1 = e1 - n_bd * np.sum(n_bd * e1, axis=1, keepdims=True)  # (n, dim)
    e1 = e1 / np.clip(
        np.linalg.norm(e1, axis=1, keepdims=True), a_min=1.0e-6, a_max=None
    )  # (n, dim)

    unit_r = unit_r.reshape(-1, dim, 1)  # (n, dim, 1)
    e1 = e1.reshape(-1, dim, 1)  # (n, dim, 1)

    if dim == 2:
        E = np.concatenate([unit_r, e1], axis=-1)
    elif dim == 3:
        e2 = np.cross(n_bd, e1.reshape(-1, dim), axis=1).reshape(-1, dim, 1)  # (n, dim)
        E = np.concatenate([unit_r, e1, e2], axis=-1)  # (n, dim)
    inv_E = np.linalg.inv(E)  # (n, dim, dim)

    if not repulsion:
        lambda_r = 1 - 1 / Gamma_o ** (1 / rho)  # (n, 1)
    else:
        lambda_r = np.zeros((f.shape[0], 1))  # (n, 1)
        inner_prod = np.sum(f * unit_r.reshape(-1, dim), axis=1)
        lambda_r[inner_prod >= 0] = 1
        lambda_r[inner_prod < 0] = 1 - (c_rep / Gamma_o[inner_prod < 0]) ** (1 / rho)

    lambda_e = 1 + 1 / Gamma_o ** (1 / rho)  # (n, 1)
    if dim == 2:
        D_diag = np.hstack([lambda_r, lambda_e])
    elif dim == 3:
        D_diag = np.hstack([lambda_r, lambda_e, lambda_e])
    # (n, 2) to diagonal matrix (n, dim, dim)
    D = np.zeros((f.shape[0], dim, dim))
    i = np.arange(dim)
    D[:, i, i] = D_diag
    M = E @ D @ inv_E
    f_modulated = (M @ f.reshape(-1, dim, 1)).reshape(-1, dim)

    if surface_friction:
        # Apply surface friction
        lambda_f = 1 - 1 / Gamma_o
        f_norm = np.linalg.norm(f_modulated, axis=1, keepdims=True)
        f_modulated_norm = np.linalg.norm(f_modulated, axis=1, keepdims=True)
        f_modulated = lambda_f * f_norm / f_modulated_norm * f_modulated

    if output_Gamma_o:
        return f_modulated, Gamma_o
    else:
        return f_modulated
