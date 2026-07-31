"""Distance-based hand–object contact detection (CPU, no Warp).

Uses trimesh proximity queries in the object CAD frame. Compatible with the
``detect_contacts`` interface used by ReactiveGrasp's UDF path
(``scripts/planning/offline/mj_eval/udf_contact_detector.py``).
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

    def query(self, points: np.ndarray, with_sign: bool = False):
        """Query distances and outward unit normals (UDFGrid-compatible).

        Parameters
        ----------
        points : (M, 3) in object/CAD frame
        with_sign : if True, also return per-point sign (+1 outside / -1 inside)

        Returns
        -------
        distances : (M,)  — unsigned distance to surface (matches UDFGrid)
        normals : (M, 3)  — unit vectors pointing outward (into free space)
        signs : (M,) — only when with_sign=True
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            empty_d = np.zeros(0)
            empty_n = np.zeros((0, 3))
            if with_sign:
                return empty_d, empty_n, np.zeros(0)
            return empty_d, empty_n

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
            flip[outside] = (
                np.sum((pts[outside] - closest[outside]) * normals[outside], axis=1)
                < 0
            )
            flip[~outside] = (
                np.sum(
                    (pts[~outside] - closest[~outside]) * normals[~outside], axis=1
                )
                > 0
            )
            normals[flip] *= -1.0
            # Return unsigned distances to match UDFGrid.query.
            distances = np.abs(signed)
            signs = np.where(signed >= 0.0, 1.0, -1.0)
            if with_sign:
                return distances, normals, signs
            return distances, normals

        # Non-watertight: unsigned distance; (p - closest) is outward when outside.
        delta = pts - closest
        d_n = np.linalg.norm(delta, axis=1, keepdims=True)
        outward = delta / np.maximum(d_n, 1e-8)
        flip = np.sum(normals * outward, axis=1) < 0
        normals[flip] *= -1.0
        distances = np.asarray(unsigned, dtype=np.float64)
        if with_sign:
            return distances, normals, np.ones(len(pts), dtype=np.float64)
        return distances, normals


def detect_contacts(
    dist_field: MeshDistanceField,
    hand_points_per_link,
    tf_link_to_obj,
    dist_threshold,
    max_contacts_per_link=1,
    hand_normals_per_link=None,
    return_signs=False,
):
    """Detect contacts between hand link surface samples and the object.

    Matches ReactiveGrasp ``udf_contact_detector.detect_contacts`` return
    conventions so ``MjEvalGraspControl._rt_detect_contacts`` can be ported
    faithfully:

    - ``contact_points_obj`` is the *hand* FK sample in object frame (not a
      mesh projection).
    - ``contact_normals_obj`` are inward (squeeze-side) normals = −outward
      field gradient (same as −∇UDF).
    - Distances are unsigned.
    - Optional hand normals / signs appended like the UDF API.

    Returns
    -------
    contact_link_indices : (K,)
    contact_points_obj : (K, 3)  object/CAD frame (hand FK samples)
    contact_normals_obj : (K, 3) inward normals in object frame
    contact_points_link : (K, 3) link frame
    contact_distances : (K,) unsigned
    contact_hand_normals_obj : (K, 3) — if hand_normals_per_link provided
    contact_signs : (K,) — if return_signs=True (+1 outside / −1 inside)
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
        pts_link = hand_points_per_link[link_idx]
        T = tf_link_to_obj[link_idx]
        R = T[:3, :3]
        pts_obj = (R @ pts_link.T).T + T[:3, 3]

        if return_signs:
            dists, normals, signs_link = dist_field.query(pts_obj, with_sign=True)
        else:
            dists, normals = dist_field.query(pts_obj)

        # Top-K closest within threshold (unsigned distances).
        order = np.argsort(dists)
        selected = []
        for idx in order:
            if dists[idx] >= dist_threshold:
                break
            selected.append(idx)
            if len(selected) >= max_contacts_per_link:
                break

        if len(selected) == 0:
            continue

        sel = np.asarray(selected, dtype=int)
        contact_link_indices.append(np.full(len(sel), link_idx, dtype=np.int64))
        contact_points_obj.append(pts_obj[sel])
        # Outward field normal → inward (squeeze) = -normal  (matches −∇UDF).
        contact_normals_obj.append(-normals[sel])
        contact_points_link.append(pts_link[sel])
        contact_distances.append(dists[sel])
        if have_hand_normals:
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
