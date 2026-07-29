import numpy as np
from scipy.spatial.transform import Rotation


def get_circular_grasp_points(center, normal, radius, num_points, grasp_width=0.03):
    # center: 3 (center of the circle)
    # normal: 3 (normal of the plane that circle is in)
    # radius: float (radius of the circle)
    # num_points: int (number of points to sample)

    grasp_points = np.zeros((num_points, 2, 3))

    # sample points on the circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    points = np.zeros((num_points, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = 0.0

    # rotate points to the plane
    # rot matrix from normal to z-axis
    rot_mat = np.eye(3)
    rot_mat[2, :] = normal
    rot_mat[0, :] = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    rot_mat[0, :] = rot_mat[0, :] / np.linalg.norm(rot_mat[0, :])
    rot_mat[1, :] = np.cross(normal, rot_mat[0, :])
    rot_mat[1, :] = rot_mat[1, :] / np.linalg.norm(rot_mat[1, :])

    points = points @ rot_mat
    points += center.reshape(1, 3)

    grasp_points[:, 0, :] = points

    unit_dir = center.reshape(1, 3) - points
    unit_dir /= np.linalg.norm(unit_dir, axis=1, keepdims=True)
    grasp_points[:, 1, :] = points + unit_dir * grasp_width

    return grasp_points  # (num_points, 2, 3)


def get_grasp_points():
    center = np.array([0.0, 0.02, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    radius = 0.05
    num_points = 20
    grasp_width = 0.02
    return get_circular_grasp_points(center, normal, radius, num_points, grasp_width)


def get_nominal_pose2bb():
    euler_xyz = np.array([0.0, 0.0, 0.0])
    trans = np.array([0.0, 0.0, 0.0])
    rot = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T
