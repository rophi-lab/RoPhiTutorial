import numpy as np

# trnasformtion matrix "sensor2dip_tip"


# spherical
def get_spherical_sensor2diptip_2G_left():
    T = np.eye(4)
    R = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
    T[:3, :3] = R
    return T


def get_spherical_sensor2diptip_2G_right():
    T = np.eye(4)
    R = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    T[:3, :3] = R
    return T


# ellipsoidal_v1
def get_ellipsoidal_sensor2diptip_2G_left():
    T = np.eye(4)
    R = np.array(
        [
            [-0.9908894, 0.0000000, 0.1346780],
            [-0.1346780, -0.0000000, -0.9908894],
            [0.0000000, -1.0000000, 0.0000000],
        ]
    )
    T[:3, :3] = R
    return T


def get_ellipsoidal_sensor2diptip_2G_right():
    T = np.eye(4)
    R = np.array(
        [
            [-0.9908894, 0.0000000, 0.1346780],
            [0.1346780, -0.0000000, 0.9908894],
            [0.0000000, 1.0000000, 0.0000000],
        ]
    )
    T[:3, :3] = R
    return T


# ellipsoidal_v2_in_new_three_fingers
def get_ellipsoidal_sensor2diptip_3G_right_sup():
    T = np.eye(4)
    R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    p = np.array([0.0145, 0.009, 0.00635])
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def get_ellipsoidal_sensor2diptip_3G_right_inf():
    return get_ellipsoidal_sensor2diptip_3G_right_sup()


########################
# NOTE: This is for mujoco
# MUJOCO uses GEOM frame,
def get_r_s_dip_tip_force_offset():
    # geom pose to sensor frame
    # NOTE: geom frame is different from the body frame
    T = np.eye(4)
    R = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    T[:3, :3] = R
    return T


def get_r_i_dip_tip_force_offset():
    return get_r_s_dip_tip_force_offset()
