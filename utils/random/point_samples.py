import numpy as np


def sample_points_from_plane(center, xhat, yhat, xrange, yrange, n_points):
    # Temporally disabled
    # x = np.random.uniform(-xrange, xrange, n_points).reshape(-1, 1)
    # y = np.random.uniform(-yrange, yrange, n_points).reshape(-1, 1)
    # return x * xhat.reshape(1, 3) + y * yhat.reshape(1, 3) + center.reshape(1, 3)

    # Sample only the center
    return center.reshape(1, 3)


def sample_points_from_box(pos, R, size, n_points):
    neg_x_surf = sample_points_from_plane(
        pos - R[:, 0] * size[0], R[:, 1], R[:, 2], size[1], size[2], n_points
    )
    pos_x_surf = sample_points_from_plane(
        pos + R[:, 0] * size[0], R[:, 1], R[:, 2], size[1], size[2], n_points
    )
    neg_y_surf = sample_points_from_plane(
        pos - R[:, 1] * size[1], R[:, 0], R[:, 2], size[0], size[2], n_points
    )
    pos_y_surf = sample_points_from_plane(
        pos + R[:, 1] * size[1], R[:, 0], R[:, 2], size[0], size[2], n_points
    )
    neg_z_surf = sample_points_from_plane(
        pos - R[:, 2] * size[2], R[:, 0], R[:, 1], size[0], size[1], n_points
    )
    pos_z_surf = sample_points_from_plane(
        pos + R[:, 2] * size[2], R[:, 0], R[:, 1], size[0], size[1], n_points
    )
    return neg_x_surf, pos_x_surf, neg_y_surf, pos_y_surf, neg_z_surf, pos_z_surf
