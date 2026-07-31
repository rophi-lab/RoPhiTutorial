"""
Sample points on link surfaces from a robot URDF for contact point generation.
Uses collision geometry for each link (no name-based matching).
"""

import os
import numpy as np
import trimesh


def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Greedily select n_samples points with maximum mutual distance (FPS).

    Args:
        points: (N, 3) candidate points (N >> n_samples)
        n_samples: number of points to select

    Returns:
        (n_samples, 3) uniformly spaced subset of points
    """
    N = len(points)
    if N <= n_samples:
        return points

    selected = np.zeros(n_samples, dtype=np.int64)
    min_dists = np.full(N, np.inf)

    # Start from a random point
    selected[0] = np.random.randint(N)

    for i in range(1, n_samples):
        last = points[selected[i - 1]]
        dists = np.sum((points - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected[i] = np.argmax(min_dists)

    return points[selected]


def _farthest_point_sample_indices(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Like _farthest_point_sample but returns indices into the input array."""
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


def _collision_geom_to_trimesh(geom, origin, mesh_dir: str):
    """Convert a collision geometry to trimesh in link frame."""
    T = np.eye(4) if origin is None else np.array(origin, dtype=np.float64)

    if geom.box is not None:
        extents = np.array(geom.box.size, dtype=np.float64)
        mesh = trimesh.creation.box(extents=extents)
    elif geom.cylinder is not None:
        r, h = geom.cylinder.radius, geom.cylinder.length
        mesh = trimesh.creation.cylinder(radius=r, height=h)
    elif geom.sphere is not None:
        r = geom.sphere.radius
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=r)
    elif geom.mesh is not None:
        fn = geom.mesh.filename
        if fn.startswith("package://"):
            parts = fn.split("package://", 1)[-1].split("/")
            fn = "/".join(parts[1:]) if len(parts) > 1 else fn
        path = os.path.join(mesh_dir, fn)
        if not os.path.isfile(path):
            path = os.path.join(mesh_dir, os.path.basename(fn))
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
    urdf_path: str, mesh_dir: str, link_names: list,
    n_samples: int = None, density_per_cm2: float = None,
    keep_normal_dirs: dict = None, normal_dot_threshold: float = 0.3,
    keep_position_filters: dict = None,
    min_samples: int = 5,
    return_normals: bool = False,
):
    """
    Load the hand URDF and sample points on each link's collision surface.

    Uses collision geometry for each link (structure-based, no name matching).

    Two sampling modes:
      - Fixed count (`n_samples`): each link gets the same number of points.
        Uses trimesh area-weighted random sampling + FPS thinning.
      - Uniform density (`density_per_cm2`): per-link target count is
        ceil(area_link * density), and points are placed via Open3D Poisson-disk
        sampling on the mesh (blue-noise distribution, near-uniform spacing).
        Recommended for force-closure / contact detection — naturally consistent
        density across the whole hand.

    Args:
        urdf_path: Path to the hand URDF file
        mesh_dir: Directory for mesh assets (same as used by URDF loader)
        link_names: List of link names in order [link_0, link_1, ...]
        n_samples: Fixed number of points per link (legacy mode).
        density_per_cm2: Target surface density in points per cm². Mutually
            exclusive with n_samples.
        keep_normal_dirs: Optional dict {link_name: direction (3,)} restricting
            samples to faces whose outward normal aligns with the given direction
            (in link frame). Used to skip back/sides of e.g. the palm.
        normal_dot_threshold: Minimum dot(face_normal, direction) for a face to
            be kept. Default 0.3 includes slightly slanted edges around the pad.
        keep_position_filters: Optional dict {link_name: (axis (3,), min_value)}.
            Additional filter on the sampled point's link-frame position: keep
            only points where (point · axis) >= min_value. Useful when the mesh
            has internal cavities so a normal filter alone is not enough (e.g.
            +Y-facing faces that are nonetheless on the back of the palm).
            Applied on top of any normal filter.
        min_samples: Floor on per-link sample count in density mode (so very
            small links still get a few points).
        return_normals: If True, also return per-link face-normal arrays in
            link frame. Useful for hand-side contact normals (independent of
            object pose error).

    Returns:
        samples_per_link: List of length len(link_names). Each element is (N, 3) array
            of points in that link's frame. N may differ across links in
            density mode.
        normals_per_link: (only if return_normals=True) List of (N, 3) face
            normals in link frame, aligned with samples_per_link.

    Raises:
        ValueError: If a link in link_names has no collision geometry, or
            neither n_samples nor density_per_cm2 is provided.
    """
    if (n_samples is None) == (density_per_cm2 is None):
        raise ValueError(
            "specify exactly one of n_samples or density_per_cm2"
        )
    from yourdfpy import URDF

    urdf = URDF.load(
        urdf_path,
        mesh_dir=mesh_dir,
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
            mesh = _collision_geom_to_trimesh(coll.geometry, origin, mesh_dir)
            if (
                mesh is not None
                and isinstance(mesh, trimesh.Trimesh)
                and len(mesh.vertices) >= 4
            ):
                meshes.append(mesh)

        if not meshes:
            raise ValueError(f"Link '{link_name}' has no collision geometry")

        combined = trimesh.util.concatenate(meshes)

        keep_dir = None
        if keep_normal_dirs is not None and link_name in keep_normal_dirs:
            keep_dir = np.asarray(keep_normal_dirs[link_name], dtype=np.float64)
            keep_dir = keep_dir / max(np.linalg.norm(keep_dir), 1e-12)

        pos_axis, pos_min = None, None
        if keep_position_filters is not None and link_name in keep_position_filters:
            pos_axis, pos_min = keep_position_filters[link_name]
            pos_axis = np.asarray(pos_axis, dtype=np.float64)
            pos_axis = pos_axis / max(np.linalg.norm(pos_axis), 1e-12)

        if density_per_cm2 is not None:
            # Density mode: oversample uniformly (area-weighted), apply filters,
            # then FPS-thin to the target count for blue-noise spacing.
            # Open3D's `sample_points_poisson_disk` was found to produce nearly
            # random distributions on this URDF's collision meshes (CV ≈ 0.7
            # for nn-distance), so we use trimesh + FPS instead.
            area_cm2 = combined.area * 1e4
            n_target = max(min_samples, int(np.ceil(area_cm2 * density_per_cm2)))

            # Oversample by 8× before filtering+FPS so the kept set is dense.
            oversample = max(n_target * 8, 2000)
            pts, face_idx = trimesh.sample.sample_surface(combined, oversample)
            normals = combined.face_normals[face_idx]

            if keep_dir is not None or pos_axis is not None:
                mask = np.ones(len(pts), dtype=bool)
                if keep_dir is not None:
                    mask &= (normals @ keep_dir) >= normal_dot_threshold
                if pos_axis is not None:
                    mask &= (pts @ pos_axis) >= pos_min
                pts = pts[mask]
                normals = normals[mask]
                # Density mode keeps target density on the filtered region:
                #   n_kept ≈ n_target × (kept_area / full_area)
                # so pull the FPS target down to match.
                n_kept_target = max(min_samples,
                                    int(np.ceil(len(pts) / 8.0)))
            else:
                n_kept_target = n_target

            if len(pts) > n_kept_target:
                idx = _farthest_point_sample_indices(pts, n_kept_target)
                pts = pts[idx]
                normals = normals[idx]

            samples_per_link.append(pts.astype(np.float64))
            normals_per_link.append(normals.astype(np.float64))
            continue

        # Legacy: fixed-count, area-weighted random + FPS.
        if keep_dir is None and pos_axis is None:
            oversample = max(n_samples * 10, 1000)
            pts, face_idx = trimesh.sample.sample_surface(combined, oversample)
            normals = combined.face_normals[face_idx]
        else:
            # Oversample more since filters can drop a large fraction.
            oversample = max(n_samples * 40, 4000)
            pts, face_idx = trimesh.sample.sample_surface(combined, oversample)
            normals = combined.face_normals[face_idx]
            mask = np.ones(len(pts), dtype=bool)
            if keep_dir is not None:
                mask &= (normals @ keep_dir) >= normal_dot_threshold
            if pos_axis is not None:
                mask &= (pts @ pos_axis) >= pos_min
            pts = pts[mask]
            normals = normals[mask]
            if len(pts) < n_samples:
                raise ValueError(
                    f"Link '{link_name}': only {len(pts)} pts pass the "
                    f"filter (need {n_samples}). Loosen the threshold or "
                    f"increase oversample."
                )

        idx = _farthest_point_sample_indices(pts, n_samples)
        samples_per_link.append(pts[idx].astype(np.float64))
        normals_per_link.append(normals[idx].astype(np.float64))

    if return_normals:
        return samples_per_link, normals_per_link
    return samples_per_link
