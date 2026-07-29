import fcl
import numpy as np


def get_fcl_geom(type: str, size: np.ndarray, *args, **kwargs):
    """
    Get the FCL geometry based on the type and size.
    @param[in] type: The type of the geometry.
    @param[in] size: The size of the geometry.
    @return: The FCL geometry.
    """
    if type == "box":
        return fcl.Box(size[0], size[1], size[2])
    elif type == "cylinder":
        return fcl.Cylinder(size[0], size[1])
    elif type == "sphere":
        return fcl.Sphere(size[0])
    else:
        raise ValueError(f"Unknown geometry type: {type}")
