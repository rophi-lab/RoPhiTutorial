import numpy as np
from scipy.spatial.transform import Rotation


def get_rect_boundary_points(axis1_max, axis2_max, num_axis1_points, num_axis2_points):
    list_points = []
    for axis1_idx in range(num_axis1_points):
        for axis2_val in [axis2_max, -axis2_max]:
            if num_axis1_points == 1:
                list_points.append(
                    [
                        [0, axis2_val],
                        [0, axis2_val],
                    ]
                )
            else:
                list_points.append(
                    [
                        [
                            -axis1_max
                            + axis1_idx * 2 * axis1_max / (num_axis1_points - 1),
                            axis2_val,
                        ],
                        [
                            -axis1_max
                            + axis1_idx * 2 * axis1_max / (num_axis1_points - 1),
                            axis2_val,
                        ],
                    ]
                )
    for axis2_idx in range(num_axis2_points):
        for axis1_val in [axis1_max, -axis1_max]:
            if num_axis2_points == 1:
                list_points.append(
                    [
                        [axis1_val, 0],
                        [axis1_val, 0],
                    ]
                )
            else:
                list_points.append(
                    [
                        [
                            axis1_val,
                            -axis2_max
                            + axis2_idx * 2 * axis2_max / (num_axis2_points - 1),
                        ],
                        [
                            axis1_val,
                            -axis2_max
                            + axis2_idx * 2 * axis2_max / (num_axis2_points - 1),
                        ],
                    ]
                )
    return np.array(list_points)


def get_grasp_points():
    # the longest axis is in the x-axis
    # the shortest axis is in the z-axis
    # the middle axis is in the y-axis

    x_half_width = 0.04
    y_half_width = 0.065
    z_half_width = 0.085

    # x-axis
    num_x_points = 2  # 4
    num_y_points = 2  # 6
    num_z_points = 2  # 8
    x_max = x_half_width - 0.04
    y_max = y_half_width - 0.065
    z_max = z_half_width - 0.085

    x_rect_boundary_points = np.zeros((2 * (num_y_points + num_z_points), 2, 3))
    x_rect_boundary_points[:, :, 1:] = get_rect_boundary_points(
        y_max, z_max, num_y_points, num_z_points
    )
    x_rect_boundary_points[:, 0, 0] = x_half_width
    x_rect_boundary_points[:, 1, 0] = -x_half_width

    y_rect_boundary_points = np.zeros((2 * (num_x_points + num_z_points), 2, 3))
    y_rect_boundary_points[:, :, [0, 2]] = get_rect_boundary_points(
        x_max, z_max, num_x_points, num_z_points
    )
    y_rect_boundary_points[:, 0, 1] = y_half_width
    y_rect_boundary_points[:, 1, 1] = -y_half_width

    z_rect_boundary_points = np.zeros((2 * (num_x_points + num_y_points), 2, 3))
    z_rect_boundary_points[:, :, :2] = get_rect_boundary_points(
        x_max, y_max, num_x_points, num_y_points
    )
    z_rect_boundary_points[:, 0, 2] = z_half_width
    z_rect_boundary_points[:, 1, 2] = -z_half_width

    return np.concatenate(
        [x_rect_boundary_points, y_rect_boundary_points, z_rect_boundary_points], axis=0
    )


def get_nominal_pose2bb():
    euler_xyz = np.array([0.0, 0.0, 0.0])
    trans = np.array([0.0, 0.0, 0.0])
    rot = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T
