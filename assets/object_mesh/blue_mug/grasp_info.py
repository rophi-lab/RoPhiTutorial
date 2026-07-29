import numpy as np
from scipy.spatial.transform import Rotation


def get_circular_grasp_points_with_handle_exclusion(
    center,
    normal,
    radius,
    num_points,
    grasp_width=0.03,
    handle_dir=np.array([-0.5, 0.0, -0.5]),
    handle_half_angle_rad=0.3,
):
    # center: 3 (center of the circle)
    # normal: 3 (normal of the plane that circle is in)
    # radius: float (radius of the circle)
    # num_points: int (number of points to sample)

    grasp_points = np.zeros((num_points, 2, 3))

    # sample points on the circle
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.zeros((num_points, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = 0.0

    # rotate points to the plane with given normal by constructing an ONB
    normal = np.asarray(normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("normal vector must be non-zero")
    normal_unit = normal / normal_norm

    # choose a reference axis not parallel to normal
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.abs(np.dot(ref, normal_unit)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)

    tangent0 = np.cross(ref, normal_unit)
    t0_norm = np.linalg.norm(tangent0)
    if t0_norm == 0:
        # fallback in pathological case
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent0 = np.cross(ref, normal_unit)
        t0_norm = np.linalg.norm(tangent0)
        if t0_norm == 0:
            tangent0 = np.array([1.0, 0.0, 0.0], dtype=float)
            t0_norm = 1.0
    tangent0 = tangent0 / t0_norm
    tangent1 = np.cross(normal_unit, tangent0)
    tangent1 = tangent1 / np.linalg.norm(tangent1)

    rot_mat = np.stack([tangent0, tangent1, normal_unit], axis=1)  # 3x3
    points = points @ rot_mat  # (N,3)
    points += center.reshape(1, 3)

    # optionally remove points near the handle direction within an angular margin
    if handle_dir is not None and handle_half_angle_rad is not None:
        handle_dir = np.asarray(handle_dir, dtype=float)
        # project handle_dir onto the plane defined by normal
        handle_proj = handle_dir - np.dot(handle_dir, normal_unit) * normal_unit
        if np.linalg.norm(handle_proj) > 1e-8:
            handle_proj /= np.linalg.norm(handle_proj)
            # compute direction from center to each point
            radial_dirs = points - center.reshape(1, 3)
            radial_dirs /= np.linalg.norm(radial_dirs, axis=1, keepdims=True)
            cos_angles = np.clip(
                np.einsum("ij,j->i", radial_dirs, handle_proj), -1.0, 1.0
            )
            # keep points whose angle from handle direction exceeds the exclusion band
            mask = np.abs(np.arccos(cos_angles)) > handle_half_angle_rad
            if np.any(mask):
                points = points[mask]
                # adjust grasp_points container size accordingly
                grasp_points = np.zeros((points.shape[0], 2, 3))

    grasp_points[:, 0, :] = points

    unit_dir = center.reshape(1, 3) - points
    unit_dir /= np.linalg.norm(unit_dir, axis=1, keepdims=True)
    grasp_points[:, 1, :] = points + unit_dir * grasp_width

    return grasp_points  # (num_points, 2, 3)


def get_grasp_points():
    center = np.array([0.0, 0.02, 0.01])
    normal = np.array([0.0, 1.0, 0.0])
    radius = 0.06
    num_points = 20
    grasp_width = 0.04
    # Exclude grasp points around the mug handle (assume +x direction in world)
    handle_dir = np.array([-0.4, 0.0, -1.0])
    handle_half_angle_rad = 0.9  # ~34 degrees
    return get_circular_grasp_points_with_handle_exclusion(
        center,
        normal,
        radius,
        num_points,
        grasp_width,
        handle_dir=handle_dir,
        handle_half_angle_rad=handle_half_angle_rad,
    )


def get_nominal_pose2bb():
    euler_xyz = np.array([0.0, 0.0, 0.0])
    trans = np.array([0.0, 0.0, 0.0])
    rot = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T
