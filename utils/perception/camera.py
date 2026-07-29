import numpy as np
from scipy.ndimage import uniform_filter
import cv2


def pixel_to_camera(
    u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """
    Convert pixel coordinates to camera coordinates using the camera intrinsic parameters.
    x = (x_pixel - cx) * z / fx
    y = (y_pixel - cy) * z / fy
    z = z
    """

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z])


def pixel_to_camera_with_K(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """
    Convert pixel coordinates to camera coordinates using the camera intrinsic matrix K.
    x = (x_pixel - cx) * z / fx
    y = (y_pixel - cy) * z / fy
    z = z
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    return pixel_to_camera(u, v, z, fx, fy, cx, cy)


def convert_pixel_to_world(
    pixel,
    depth_image,
    cam_intrinsic,
    cam2world,
    depth_factor=1.0,
    inverse_z_direction=False,
):
    """
    Convert pixel coordinates to world coordinates.
    Args:
        pixel (tuple): Pixel coordinates (x, y).
        depth_image (numpy.ndarray): Depth image.
        cam_intrinsic (numpy.ndarray): Camera intrinsic matrix.
        cam2world (numpy.ndarray): Camera to world transformation matrix.
    """

    z = depth_image[int(pixel[1]), int(pixel[0])]

    # filter out invalid depth values
    if z <= 1 or np.isnan(z):
        return None  # or raise an error / skip

    z = -z / depth_factor if inverse_z_direction else z / depth_factor

    p_camera = pixel_to_camera_with_K(pixel[0], pixel[1], z, cam_intrinsic)
    return cam2world[:3, :3] @ p_camera + cam2world[:3, 3]


def fill_depth_mean(depth_image, window):
    """
    Fill invalid depths (NaN or <= 1) using local mean of valid neighbors via box filtering.
    No Python loops.
    """
    assert window % 2 == 1 and window >= 3, "fill_window must be an odd integer >= 3"
    H, W = depth_image.shape[:2]

    # Valid if finite and > 1 (keep your original criterion)
    valid = np.isfinite(depth_image) & (depth_image > 1.0)

    # If SciPy is available, use uniform_filter (very fast)

    area = float(window * window)

    depth_zeroed = np.where(valid, depth_image, 0.0)

    mean_depth = uniform_filter(depth_zeroed, size=window, mode="nearest")
    mean_mask = uniform_filter(valid.astype(np.float32), size=window, mode="nearest")

    # Convert uniform means back to sums/counts
    sum_depth = mean_depth * area
    count = mean_mask * area

    # Local mean of valid neighbors
    local_mean = np.divide(
        sum_depth, count, out=np.full_like(sum_depth, np.nan), where=count > 0
    )

    # Use original where valid; otherwise use local mean if it exists
    depth_filled = np.where(valid, depth_image, local_mean)
    return depth_filled


def fill_depth_median(depth_image, window):
    """
    Fill invalid depths using local median (no Python loops).
    Implemented via OpenCV medianBlur (fast). Requires integer depths or conversion.
    """
    assert window % 2 == 1 and window >= 3, "fill_window must be an odd integer >= 3"

    # OpenCV medianBlur expects 8/16-bit or float; but NaNs propagate poorly.
    # Strategy:
    #   1) Replace invalid with 0 temporarily
    #   2) Median blur
    #   3) Put back originals where valid, otherwise use median result (if > 0)
    valid = np.isfinite(depth_image) & (depth_image > 1.0)

    # Choose a dtype OpenCV handles well. If your depth is uint16 (RealSense), this is perfect.
    if depth_image.dtype != np.uint16:
        # Scale to uint16 safely (assumes depth range reasonable; adjust scale as needed)
        scale = 1.0
        di = depth_image.copy().astype(np.float32)
        di[~valid] = 0.0
        med = cv2.medianBlur(di, ksize=window)
        # Use median where invalid and median > 0
        filled = np.where(valid, depth_image, np.where(med > 0, med, np.nan))
        return filled.astype(depth_image.dtype)
    else:
        di = depth_image.copy()
        di[~valid] = 0  # invalid → 0
        med = cv2.medianBlur(di, ksize=window)
        # If median is zero, keep NaN; else use median
        filled = np.where(valid, depth_image, np.where(med > 0, med, np.nan))
        return filled.astype(depth_image.dtype)


def convert_pixel_to_world_batch(
    pixel,
    depth_image,
    cam_intrinsic,
    cam2world,
    depth_factor=1.0,
    inverse_z_direction=False,
    remove_invalid=False,
    fill_invalid_depth=False,
    fill_window=5,
    fill_method="mean",  # "mean" (box filter) or "median" (OpenCV)
):
    """
    Convert pixel coordinates to world coordinates with optional neighbor-based depth filling.

    Args:
        pixel (tuple or np.ndarray): Either a single (x, y) or an array of shape (N, 2).
        depth_image (np.ndarray): (H, W) raw depth image.
        cam_intrinsic (np.ndarray): (3, 3) intrinsics.
        cam2world (np.ndarray): (4, 4) camera-to-world transform.
        depth_factor (float): depth scale (e.g., 1000 for mm->m).
        inverse_z_direction (bool): flip Z sign if camera forward is -Z.
        remove_invalid (bool): if True, drop invalid rows; else keep NaNs, preserving shape.
        fill_invalid_depth (bool): if True, fill invalid depths from neighbors (no loops).
        fill_window (int): odd window size for neighborhood (>=3).
        fill_method (str): "mean" (fast, SciPy/OpenCV) or "median" (OpenCV).

    Returns:
        world_pts: (N, 3) if remove_invalid=False; else (M, 3) with M <= N.
        valid_mask: (N,) boolean validity mask (post-fill).
    """
    # Normalize input to (N, 2)
    pixels = np.asarray(pixel)
    if pixels.ndim == 1:
        pixels = pixels[None, :]

    H, W = depth_image.shape[:2]
    N = pixels.shape[0]

    # Optionally fill invalid depths (vectorized, no loops)
    if fill_invalid_depth:
        if fill_method == "mean":
            depth_used = fill_depth_mean(depth_image, fill_window)
        elif fill_method == "median":
            depth_used = fill_depth_median(depth_image, fill_window)
        else:
            raise ValueError("fill_method must be 'mean' or 'median'")
    else:
        depth_used = depth_image

    fx, fy = cam_intrinsic[0, 0], cam_intrinsic[1, 1]
    cx, cy = cam_intrinsic[0, 2], cam_intrinsic[1, 2]

    # Integer indices
    xs = pixels[:, 0].astype(int)
    ys = pixels[:, 1].astype(int)

    # Bounds
    in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

    # Sample depths
    z_raw = np.full(N, np.nan, dtype=float)
    z_raw[in_bounds] = depth_used[ys[in_bounds], xs[in_bounds]].astype(float)

    # Validity: your original rule (before scaling)
    valid_depth = np.isfinite(z_raw) & (z_raw > 1.0)

    # Scale and (optionally) flip Z
    z = z_raw / float(depth_factor)
    if inverse_z_direction:
        z = -z

    # Final validity mask
    valid = in_bounds & valid_depth & np.isfinite(z)

    # Prepare output
    world_pts = np.full((N, 3), np.nan, dtype=float)

    if np.any(valid):
        u = xs[valid].astype(float)
        v = ys[valid].astype(float)
        zv = z[valid]

        Xc = (u - cx) * zv / fx
        Yc = (v - cy) * zv / fy
        Zc = zv

        Pc = np.stack([Xc, Yc, Zc], axis=1)  # (M, 3)
        R = cam2world[:3, :3]
        t = cam2world[:3, 3]
        world_pts[valid] = (R @ Pc.T).T + t

    if remove_invalid:
        return world_pts[valid], valid
    else:
        return world_pts, valid
