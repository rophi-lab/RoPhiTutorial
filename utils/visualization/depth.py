import numpy as np
import cv2


# Normalize depth image for visualization
def normalize_depth_image(depth_image):
    """
    Normalize the depth image for visualization.
    Args:
        depth_image (np.ndarray): The input depth image.
    Returns:
        np.ndarray: The normalized depth image.
    """
    depth_display = None
    if np.issubdtype(depth_image.dtype, np.floating):
        depth_vis = (depth_image - np.min(depth_image)) / (
            np.max(depth_image) - np.min(depth_image) + 1e-5
        )
        depth_display = (depth_vis * 255).astype(np.uint8)
    else:
        depth_display = cv2.convertScaleAbs(depth_image)
    return depth_display
