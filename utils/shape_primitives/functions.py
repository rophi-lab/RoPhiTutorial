import numpy as np


def bb2superellipsoids(x_width, y_width, z_width, margin=1.0):
    """
    Convert a bounding box to a superellipsoids.
    """
    a1 = x_width * 0.5 * margin
    a2 = y_width * 0.5 * margin
    a3 = z_width * 0.5 * margin
    e1 = 0.3
    e2 = 0.3
    return a1, a2, a3, e1, e2
