"""Distance-based hand–object contact detection (CPU, no Warp).

Uses trimesh proximity queries in the object CAD frame. Compatible with the
``detect_contacts`` interface used by Manipulator-Software's UDF path.
"""

from __future__ import annotations

import numpy as np
import trimesh


class MeshDistanceField:
    """Nearest-surface distance + outward normal in the object CAD frame."""

    def __init__(self, mesh: trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
        self.mesh = mesh
        self._pq = trimesh.proximity.ProximityQuery(mesh)
        self._watertight = bool(mesh.is_watertight)

    def query(self, points: np.ndarray):
        """Query unsigned/signed distances and outward unit normals.

        Parameters
        ----------
        points : (M, 3) in object/CAD frame

        Returns
        -------
        distances : (M,)  — signed if watertight (neg inside), else unsigned
        normals : (M, 3)  — unit vectors pointing outward (into free space)
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return np.zeros(0), np.zeros((0, 3))

        closest, unsigned, face_id = self._pq.on_surface(pts)
        face_n = self.mesh.face_normals[face_id]

        # Prefer face normal (stable on thin walls); fall back to (p - closest).
        normals = face_n.copy()
        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        bad = nrm.ravel() < 1e-8
        if np.any(bad):
            alt = pts[bad] - closest[bad]
            alt_n = np.linalg.norm(alt, axis=1, keepdims=True)
            normals[bad] = alt / np.maximum(alt_n, 1e-8)
            nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(nrm, 1e-8)

        if self._watertight:
            signed = np.asarray(self._pq.signed_distance(pts), dtype=np.float64)
            # Orient face normals outward: outside points should have
            # signed > 0 and (p - closest) · n > 0.
            outside = signed >= 0.0
            flip = np.zeros(len(pts), dtype=bool)
            flip[outside] = np.sum((pts[outside] - closest[outside]) * normals[outside], axis=1) < 0
            flip[~outside] = np.sum((pts[~outside] - closest[~outside]) * normals[~outside], axis=1) > 0
            normals[flip] *= -1.0
            return signed, normals

        # Non-watertight: unsigned distance; (p - closest) is outward when outside.
        delta = pts - closest
        d_n = np.linalg.norm(delta, axis=1, keepdims=True)
        outward = delta / np.maximum(d_n, 1e-8)
        # Align face normal with outward when possible.
        flip = np.sum(normals * outward, axis=1) < 0
        normals[flip] *= -1.0
        return np.asarray(unsigned, dtype=np.float64), normals


def detect_contacts(
    dist_field: MeshDistanceField,
    hand_points_per_link,
    tf_link_to_obj,
    dist_threshold,
    max_contacts_per_link=1,
    hand_normals_per_link=None,
):
    """Detect contacts between hand link surface samples and the object.

    Returns
    -------
    contact_link_indices : (K,)
    contact_points_obj : (K, 3)  object/CAD frame
    contact_normals_obj : (K, 3) inward (squeeze-side) normals in object frame
    contact_points_link : (K, 3) link frame
    contact_distances : (K,)
    """
    contact_link_indices = []
    contact_points_obj = []
    contact_normals_obj = []
    contact_points_link = []
    contact_distances = []

    for link_idx in range(len(hand_points_per_link)):
        pts_link = hand_points_per_link[link_idx]
        T = tf_link_to_obj[link_idx]
        pts_obj = (T[:3, :3] @ pts_link.T).T + T[:3, 3]

        dists, normals = dist_field.query(pts_obj)

        order = np.argsort(dists)
        selected = []
        for idx in order:
            if dists[idx] >= dist_threshold:
                break
            selected.append(idx)
            if len(selected) >= max_contacts_per_link:
                break

        if not selected:
            continue

        sel = np.asarray(selected, dtype=int)
        # Outward field normal → inward (squeeze) = -normal.
        contact_n = -normals[sel]

        if hand_normals_per_link is not None:
            pen = dists[sel] < 0.0
            if np.any(pen):
                n_hand_link = hand_normals_per_link[link_idx][sel[pen]]
                n_hand_obj = (T[:3, :3] @ n_hand_link.T).T
                n_hand_obj /= np.maximum(
                    np.linalg.norm(n_hand_obj, axis=1, keepdims=True), 1e-8
                )
                contact_n[pen] = -n_hand_obj

        contact_link_indices.append(np.full(len(sel), link_idx, dtype=np.int64))
        contact_points_obj.append(pts_obj[sel])
        contact_normals_obj.append(contact_n)
        contact_points_link.append(pts_link[sel])
        contact_distances.append(dists[sel])

    if not contact_link_indices:
        return (
            np.array([], dtype=np.int64),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.array([]),
        )

    return (
        np.concatenate(contact_link_indices),
        np.concatenate(contact_points_obj),
        np.concatenate(contact_normals_obj),
        np.concatenate(contact_points_link),
        np.concatenate(contact_distances),
    )
