"""Sample points on URDF link collision surfaces for contact detection.

Ported from Manipulator-Software
``controller/brl_arm_5F_hand/grasping/reactive_grasping/utils/link_surface_sampler.py``.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh


def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    N = len(points)
    if N <= n_samples:
        return np.arange(N)

    selected = np.zeros(n_samples, dtype=np.int64)
    min_dists = np.full(N, np.inf)
    selected[0] = np.random.randint(N)
    for i in range(1, n_samples):
        last = points[selected[i - 1]]
        dists = np.sum((points - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected[i] = np.argmax(min_dists)
    return selected


def _collision_geom_to_trimesh(geom, origin, mesh_dirs: list[str], urdf_dir: str):
    T = np.eye(4) if origin is None else np.array(origin, dtype=np.float64)

    if geom.box is not None:
        mesh = trimesh.creation.box(extents=np.array(geom.box.size, dtype=np.float64))
    elif geom.cylinder is not None:
        mesh = trimesh.creation.cylinder(
            radius=geom.cylinder.radius, height=geom.cylinder.length
        )
    elif geom.sphere is not None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=geom.sphere.radius)
    elif geom.mesh is not None:
        fn = geom.mesh.filename
        if fn.startswith("package://"):
            parts = fn.split("package://", 1)[-1].split("/")
            fn = "/".join(parts[1:]) if len(parts) > 1 else fn
        path = None
        candidates = [
            os.path.normpath(os.path.join(urdf_dir, fn)),
            os.path.normpath(os.path.join(urdf_dir, os.path.basename(fn))),
        ]
        for mesh_dir in mesh_dirs:
            candidates.append(os.path.normpath(os.path.join(mesh_dir, fn)))
            candidates.append(
                os.path.normpath(os.path.join(mesh_dir, os.path.basename(fn)))
            )
            # Strip leading ../ from URDF-relative mesh paths.
            stripped = fn
            while stripped.startswith("../"):
                stripped = stripped[3:]
            candidates.append(os.path.normpath(os.path.join(mesh_dir, stripped)))
        for cand in candidates:
            if os.path.isfile(cand):
                path = cand
                break
        if path is None:
            raise FileNotFoundError(
                f"Collision mesh not found for '{fn}'. Tried:\n  "
                + "\n  ".join(candidates[:8])
            )
        loaded = trimesh.load(path, force="mesh")
        mesh = (
            loaded.dump(concatenate=True)
            if isinstance(loaded, trimesh.Scene)
            else loaded
        )
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        scale = geom.mesh.scale
        if scale is not None:
            s = np.atleast_1d(np.asarray(scale, dtype=np.float64))
            if s.size == 1:
                mesh.apply_scale(s.item())
            else:
                mesh.apply_scale(s)
    else:
        return None

    mesh.apply_transform(T)
    return mesh


def load_link_surface_samples(
    urdf_path: str,
    mesh_dir: str | list[str],
    link_names: list,
    n_samples: int = 100,
    return_normals: bool = False,
):
    """Sample collision-surface points (and optional outward normals) per link."""
    from yourdfpy import URDF

    mesh_dirs = [mesh_dir] if isinstance(mesh_dir, str) else list(mesh_dir)
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    # Prefer the first mesh dir for yourdfpy; we resolve meshes ourselves.
    urdf = URDF.load(
        urdf_path,
        mesh_dir=mesh_dirs[0],
        load_meshes=False,
        build_scene_graph=False,
        load_collision_meshes=False,
        build_collision_scene_graph=False,
        force_mesh=True,
    )

    samples_per_link = []
    normals_per_link = []
    for link_name in link_names:
        if link_name not in urdf.link_map:
            raise ValueError(f"Link '{link_name}' not found in URDF")
        link = urdf.link_map[link_name]

        meshes = []
        for coll in link.collisions:
            origin = coll.origin
            if origin is not None:
                origin = np.array(origin, dtype=np.float64)
            mesh = _collision_geom_to_trimesh(
                coll.geometry, origin, mesh_dirs, urdf_dir
            )
            if (
                mesh is not None
                and isinstance(mesh, trimesh.Trimesh)
                and len(mesh.vertices) >= 4
            ):
                meshes.append(mesh)

        if not meshes:
            raise ValueError(f"Link '{link_name}' has no collision geometry")

        combined = trimesh.util.concatenate(meshes)
        oversample = max(n_samples * 10, 1000)
        pts, face_idx = trimesh.sample.sample_surface(combined, oversample)
        sel = _farthest_point_sample(pts, n_samples)
        samples_per_link.append(pts[sel].astype(np.float64))
        if return_normals:
            normals_per_link.append(
                combined.face_normals[face_idx[sel]].astype(np.float64)
            )

    if return_normals:
        return samples_per_link, normals_per_link
    return samples_per_link
