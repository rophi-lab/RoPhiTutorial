import importlib
from pathlib import Path

import numpy as np
import trimesh


def _get_available_objects():
    """Get list of available object names by scanning directories."""
    current_dir = Path(__file__).parent
    objects = []
    for item in current_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            # Check if the directory has a grasp_info.py file
            grasp_info_file = item / "grasp_info.py"
            if grasp_info_file.exists():
                objects.append(item.name)
    return objects


def _import_object_function(object_name, function_name):
    """Dynamically import a function from an object's grasp_info module."""
    try:
        module = importlib.import_module(
            f".{object_name}.grasp_info", package=__package__
        )
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Function {function_name} not found for object {object_name}: {e}"
        )


def get_grasp_points(object_name):
    """Grasp points in the object's nominal / bounding-box frame."""
    available_objects = _get_available_objects()
    if object_name not in available_objects:
        raise ValueError(
            f"Object name '{object_name}' not found. Available objects: {available_objects}"
        )

    function = _import_object_function(object_name, "get_grasp_points")
    return function()


def get_nominal_pose2bb(object_name):
    """Get nominal pose to bounding box for the specified object."""
    available_objects = _get_available_objects()
    if object_name not in available_objects:
        raise ValueError(
            f"Object name '{object_name}' not found. Available objects: {available_objects}"
        )

    function = _import_object_function(object_name, "get_nominal_pose2bb")
    return function()


def get_nominal_pose_to_cad(object_name):
    """Map nominal/bb grasp frame → CAD / textured-mesh frame.

    Same transform as ``mesh_visualizer_with_grasp_points.py``:
    ``inv(oriented_bounds(mesh)) @ get_nominal_pose2bb(object)``.
    """
    mesh_path = Path(__file__).parent / object_name / "textured_mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh for {object_name}: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    cad_to_bb, _ = trimesh.bounds.oriented_bounds(mesh)
    return np.linalg.inv(cad_to_bb) @ get_nominal_pose2bb(object_name)


def get_grasp_points_in_cad(object_name):
    """Grasp antipodal pairs expressed in the CAD / MuJoCo mesh frame."""
    gps = np.asarray(get_grasp_points(object_name), dtype=np.float64)
    T = get_nominal_pose_to_cad(object_name)
    R, t = T[:3, :3], T[:3, 3]
    return np.einsum("ij,abj->abi", R, gps) + t.reshape(1, 1, 3)


# Optional: Provide a function to list all available objects
def list_available_objects():
    """List all available object names."""
    return _get_available_objects()
