"""
UDF-based contact detection between hand link surfaces and an object mesh.

Precomputes a 128^3 UDF grid (GPU-accelerated via Warp), then at runtime uses
trilinear interpolation for O(1) per-point distance + normal queries.
"""

import numpy as np
import trimesh
import warp as wp
from scipy.ndimage import map_coordinates


@wp.kernel
def _mesh_udf_kernel(
    mesh: wp.uint64,
    points: wp.array1d(dtype=wp.vec3),
    max_dist: float,
    udf: wp.array1d(dtype=wp.float32),
):
    tid = wp.tid()
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(
        mesh, points[tid], max_dist, sign, face_index, face_u, face_v
    )
    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        udf[tid] = wp.length(points[tid] - closest)
    else:
        udf[tid] = max_dist


@wp.kernel
def _mesh_udf_sign_kernel(
    mesh: wp.uint64,
    points: wp.array1d(dtype=wp.vec3),
    max_dist: float,
    udf: wp.array1d(dtype=wp.float32),
    sign_arr: wp.array1d(dtype=wp.float32),
):
    tid = wp.tid()
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(
        mesh, points[tid], max_dist, sign, face_index, face_u, face_v
    )
    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        udf[tid] = wp.length(points[tid] - closest)
        sign_arr[tid] = sign
    else:
        udf[tid] = max_dist
        sign_arr[tid] = 1.0


class UDFGrid:
    """Precomputed UDF grid with trilinear interpolation queries."""

    def __init__(self, mesh: trimesh.Trimesh, resolution: int = 128, padding: float = 0.02, device: str = "cuda:0"):
        wp.init()
        # Prefer the requested device; if CUDA is asked for but no CUDA device
        # is present, fall back to CPU.
        try:
            available = {str(d) for d in wp.get_devices()}
        except Exception:
            available = set()
        if device.startswith("cuda") and not any(d.startswith("cuda") for d in available):
            print(
                f"[UDFGrid] device '{device}' unavailable "
                f"(devices={sorted(available) or ['?']}), falling back to cpu"
            )
            device = "cpu"
        else:
            try:
                wp.get_device(device)
            except Exception:
                print(f"[UDFGrid] device '{device}' unavailable, falling back to cpu")
                device = "cpu"

        bb_min = mesh.vertices.min(axis=0) - padding
        bb_max = mesh.vertices.max(axis=0) + padding
        self.origin = bb_min.astype(np.float64)
        extent = bb_max - bb_min
        self.spacing = (extent / resolution).astype(np.float64)
        self.resolution = resolution
        self._device = device

        # Build grid coordinates
        axes = [np.linspace(bb_min[i], bb_max[i], resolution) for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        grid_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)

        # Create warp mesh (GPU if available, else CPU)
        warp_mesh = wp.Mesh(
            points=wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(mesh.faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
        )

        # Query UDF + sign on GPU. Sign comes from wp.mesh_query_point_sign_normal:
        # +1 outside the (closed) mesh, -1 inside.
        points_wp = wp.array(grid_points, dtype=wp.vec3, device=device)
        udf_wp = wp.empty(len(grid_points), dtype=wp.float32, device=device)
        sign_wp = wp.empty(len(grid_points), dtype=wp.float32, device=device)
        wp.launch(
            kernel=_mesh_udf_sign_kernel,
            dim=len(grid_points),
            inputs=[warp_mesh.id, points_wp, 1.0, udf_wp, sign_wp],
            device=device,
        )
        wp.synchronize_device(device)

        self.grid = udf_wp.numpy().reshape(resolution, resolution, resolution)
        self.sign_grid = sign_wp.numpy().reshape(resolution, resolution, resolution)

        # Precompute gradient grids for normals
        # np.gradient returns (d/daxis0, d/daxis1, d/daxis2) = (d/dx, d/dy, d/dz)
        self.grad_x, self.grad_y, self.grad_z = np.gradient(
            self.grid, self.spacing[0], self.spacing[1], self.spacing[2]
        )

    def query(self, points: np.ndarray, with_sign: bool = False):
        """Trilinear interpolation lookup.

        Args:
            points: (M, 3) query points in object/mesh frame
            with_sign: if True, additionally return per-point sign from the
                precomputed sign grid (+1 outside / -1 inside, nearest-voxel).
        Returns:
            distances: (M,) unsigned distances
            normals: (M, 3) UDF gradient direction (points away from surface)
            signs: (M,) — only when with_sign=True
        """
        grid_coords = (points - self.origin) / self.spacing
        grid_coords = np.clip(grid_coords, 0, self.resolution - 1.001)
        coords = grid_coords.T  # (3, M)

        distances = map_coordinates(self.grid, coords, order=1, mode="nearest")
        nx = map_coordinates(self.grad_x, coords, order=1, mode="nearest")
        ny = map_coordinates(self.grad_y, coords, order=1, mode="nearest")
        nz = map_coordinates(self.grad_z, coords, order=1, mode="nearest")
        normals = np.stack([nx, ny, nz], axis=-1)

        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)

        if not with_sign:
            return distances, normals
        # Nearest-voxel for the sign — interpolating ±1 across the surface
        # boundary doesn't give a meaningful continuous value.
        signs = map_coordinates(self.sign_grid, coords, order=0, mode="nearest")
        return distances, normals, signs


def detect_contacts(udf_grid: UDFGrid, hand_points_per_link, tf_link_to_obj, dist_threshold,
                    max_contacts_per_link=1, hand_normals_per_link=None,
                    return_signs=False):
    """Detect contacts between hand link surface points and object.

    For each link, returns the top-K closest points within the threshold.

    Note on the contact-point convention: ``contact_points_obj`` is the *hand*
    surface sample point transformed into the object/CAD frame via FK — NOT a
    mesh-projection onto the object surface. The UDF query only supplies the
    distance and the gradient (used as the default contact normal). This means
    the *point* is already pose-error-robust; only the *normal* differs between
    UDF gradient (object-derived) and hand-link face normal (hand-derived).

    Args:
        udf_grid: UDFGrid instance (object frame)
        hand_points_per_link: list of (Ni, 3) arrays in link frame
        tf_link_to_obj: list of (4,4) transforms, link -> object frame
        dist_threshold: contact distance threshold in meters
        max_contacts_per_link: max number of contact points per link
        hand_normals_per_link: optional list of (Ni, 3) face normals in link
            frame, aligned with ``hand_points_per_link``. When provided, also
            returns hand-side contact normals in object frame.
        return_signs: when True, additionally returns the per-contact sign of
            the UDF at the hand-sample position (+1 outside, -1 inside).
            Caller can use this to flip object normals where the hand sample
            has penetrated the object (∇UDF sign reverses inside).

    Returns:
        contact_link_indices: (K,) which link each contact belongs to
        contact_points_obj: (K, 3) contact points in object frame
        contact_normals_obj: (K, 3) inward normals in object frame (UDF-derived
            push direction; opposite of UDF gradient)
        contact_points_link: (K, 3) contact points in link frame
        contact_distances: (K,) unsigned distances to surface
        contact_hand_normals_obj: (K, 3) hand-side contact normals in object
            frame (only returned when hand_normals_per_link is provided). These
            are outward normals of the hand link surface, transformed via FK,
            and already point in the push direction (hand → object).
        contact_signs: (K,) signs at hand-sample positions (only returned when
            return_signs=True). Appended after hand normals if both are on.
    """
    contact_link_indices = []
    contact_points_obj = []
    contact_normals_obj = []
    contact_points_link = []
    contact_distances = []
    contact_hand_normals_obj = []
    contact_signs = []
    have_hand_normals = hand_normals_per_link is not None

    for link_idx in range(len(hand_points_per_link)):
        pts_link = hand_points_per_link[link_idx]  # (Ni, 3)
        T = tf_link_to_obj[link_idx]  # (4, 4)
        R = T[:3, :3]
        pts_obj = (R @ pts_link.T).T + T[:3, 3]  # (Ni, 3)

        if return_signs:
            dists, normals, signs_link = udf_grid.query(pts_obj, with_sign=True)
        else:
            dists, normals = udf_grid.query(pts_obj)

        # Top-K closest within threshold
        order = np.argsort(dists)
        selected = []
        for idx in order:
            if dists[idx] >= dist_threshold:
                break
            selected.append(idx)
            if len(selected) >= max_contacts_per_link:
                break

        if len(selected) > 0:
            sel = np.array(selected)
            contact_link_indices.append(np.full(len(sel), link_idx, dtype=np.int64))
            contact_points_obj.append(pts_obj[sel])
            contact_normals_obj.append(-normals[sel])
            contact_points_link.append(pts_link[sel])
            contact_distances.append(dists[sel])
            if have_hand_normals:
                # Rotate link-frame face normal to object frame (vectors only,
                # no translation). Hand outward normal already points in the
                # push direction (hand → object) for actual contacts.
                n_link = hand_normals_per_link[link_idx][sel]
                contact_hand_normals_obj.append((R @ n_link.T).T)
            if return_signs:
                contact_signs.append(signs_link[sel])

    if len(contact_link_indices) == 0:
        empty = (
            np.array([], dtype=np.int64),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.array([]),
        )
        extras = ()
        if have_hand_normals:
            extras = extras + (np.zeros((0, 3)),)
        if return_signs:
            extras = extras + (np.array([]),)
        return (*empty, *extras)

    out = (
        np.concatenate(contact_link_indices),
        np.concatenate(contact_points_obj),
        np.concatenate(contact_normals_obj),
        np.concatenate(contact_points_link),
        np.concatenate(contact_distances),
    )
    extras = ()
    if have_hand_normals:
        extras = extras + (np.concatenate(contact_hand_normals_obj),)
    if return_signs:
        extras = extras + (np.concatenate(contact_signs),)
    return (*out, *extras)
