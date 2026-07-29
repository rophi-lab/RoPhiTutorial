"""Extract static collision geometries from a compiled MuJoCo model as
ColInfoData-style box/cylinder/sphere primitives (world-frame offsets).

Single source of truth: named geoms in the scene XML. Both the collision-info
env and the viser visualizer derive static obstacles from here so they never
drift apart.

Naming conventions (exported to FCL / Viser):
- ``floor`` — plane (exported as a large thin box; FCL has no plane)
- ``wall_*`` — cell walls
- ``table`` / ``table_*`` — table surface / legs
- ``obs_*`` — tabletop / freestanding obstacles
"""
import numpy as np
import mujoco as mj

# Box used to represent the infinite floor plane (FCL/viz have no plane).
_FLOOR_BOX_HALF_THICKNESS = 0.05
_FLOOR_BOX_HALF_EXTENT = 4.0


def is_static_col_geom_name(name) -> bool:
    if name is None:
        return False
    return (
        name == "floor"
        or name.startswith("wall")
        or name == "table"
        or name.startswith("table_")
        or name.startswith("obs_")
    )


def is_wall_geom_name(name) -> bool:
    """Backward-compatible alias."""
    return is_static_col_geom_name(name)


def extract_static_col_geoms(mj_model):
    """Return dicts ``{name, geom_type, size (full), offset (4x4 world SE3)}``."""
    geoms = []
    for gid in range(mj_model.ngeom):
        name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_GEOM, gid)
        if not is_static_col_geom_name(name):
            continue
        gtype = int(mj_model.geom_type[gid])
        pos = mj_model.geom_pos[gid].copy()
        quat = mj_model.geom_quat[gid].copy()  # (w, x, y, z)
        R = np.zeros(9)
        mj.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        offset = np.eye(4)
        offset[:3, :3] = R

        if gtype == mj.mjtGeom.mjGEOM_BOX:
            size = 2.0 * mj_model.geom_size[gid].copy()  # half -> full extents
            offset[:3, 3] = pos
            geom_type = "box"
        elif gtype == mj.mjtGeom.mjGEOM_CYLINDER:
            # MuJoCo: (radius, half-height, unused) → FCL: (radius, full height)
            gs = mj_model.geom_size[gid]
            size = np.array([gs[0], 2.0 * gs[1]])
            offset[:3, 3] = pos
            geom_type = "cylinder"
        elif gtype == mj.mjtGeom.mjGEOM_SPHERE:
            size = np.array([mj_model.geom_size[gid][0]])
            offset[:3, 3] = pos
            geom_type = "sphere"
        elif gtype == mj.mjtGeom.mjGEOM_PLANE:
            size = np.array(
                [
                    2 * _FLOOR_BOX_HALF_EXTENT,
                    2 * _FLOOR_BOX_HALF_EXTENT,
                    2 * _FLOOR_BOX_HALF_THICKNESS,
                ]
            )
            # centered below the plane so its top face is at the plane height
            offset[:3, 3] = pos + R @ np.array([0.0, 0.0, -_FLOOR_BOX_HALF_THICKNESS])
            geom_type = "box"
        else:
            continue  # unsupported static shape
        geoms.append(
            {"name": name, "geom_type": geom_type, "size": size, "offset": offset}
        )
    return geoms


def extract_wall_col_geoms(mj_model):
    """Backward-compatible alias for :func:`extract_static_col_geoms`."""
    return extract_static_col_geoms(mj_model)
