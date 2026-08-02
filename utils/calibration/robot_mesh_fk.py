"""Articulate Flexiv arm + hand visual meshes via yourdfpy FK."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Set

import networkx as nx
import numpy as np
import trimesh
from yourdfpy import URDF

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_DEFAULT_URDF = os.path.join(
    _REPO_ROOT, "assets/scene/flexiv_arm/urdf/Rizon4_viser.urdf"
)

# Flat gray in [0, 255] for silhouette-style rendering.
_GRAY_RGB = np.array([180, 180, 180], dtype=np.uint8)

# Links that are *not* part of the arm/hand robot mesh (cell walls, etc.).
_DEFAULT_EXCLUDE_LINKS = frozenset({"world", "enclosure", "palm_surface"})


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


def _default_robot_links(
    link_names: Iterable[str],
    *,
    arm_only: bool = False,
    exclude_link_prefixes: Sequence[str] = (),
) -> Set[str]:
    """Arm (+ optional hand) links: base, Rizon links, mount (no enclosure).

    Matches Manipulator-Software easyhec usage of ``Rizon4_arm_only.urdf`` when
    ``arm_only=True`` (drops palm / fingers — cleaner silhouettes for calib).
    """
    keep: Set[str] = set()
    for name in link_names:
        if name in _DEFAULT_EXCLUDE_LINKS:
            continue
        if any(name.startswith(p) for p in exclude_link_prefixes):
            continue
        if name in ("base_link", "flange", "hand_mount"):
            keep.add(name)
        elif name.startswith("link") and name[4:].isdigit():
            keep.add(name)
        elif not arm_only and name == "palm":
            keep.add(name)
        elif not arm_only and name.startswith("finger_r_link_"):
            keep.add(name)
    return keep


class RobotMeshFK:
    """Load URDF visuals and return a world-frame gray trimesh at configuration q.

    Only arm + hand link meshes are concatenated (enclosure / world fixtures
    are dropped so differentiable rendering matches the robot silhouette).
    """

    def __init__(
        self,
        urdf_path: str = _DEFAULT_URDF,
        mesh_dir: Optional[str] = None,
        max_faces: int = 0,
        gray_rgb: np.ndarray = _GRAY_RGB,
        include_links: Optional[Sequence[str]] = None,
        exclude_links: Optional[Sequence[str]] = None,
        exclude_link_prefixes: Optional[Sequence[str]] = None,
        arm_only: bool = False,
    ):
        urdf_path = _resolve(urdf_path)
        kwargs = dict(
            load_meshes=True,
            build_scene_graph=True,
            load_collision_meshes=False,
            build_collision_scene_graph=False,
        )
        if mesh_dir:
            kwargs["mesh_dir"] = _resolve(mesh_dir)
        self._urdf = URDF.load(urdf_path, **kwargs)
        self._max_faces = int(max_faces)
        self._gray_rgb = np.asarray(gray_rgb, dtype=np.uint8).reshape(3)
        self.num_joints = len(self._urdf.actuated_joint_names)
        self.joint_names = list(self._urdf.actuated_joint_names)

        prefixes = tuple(exclude_link_prefixes or ())
        if include_links is not None:
            self._keep_links = set(include_links)
        else:
            self._keep_links = _default_robot_links(
                self._urdf.link_map.keys(),
                arm_only=bool(arm_only),
                exclude_link_prefixes=prefixes,
            )
        if exclude_links:
            self._keep_links -= set(exclude_links)
        print(
            f"[RobotMeshFK] Keeping {len(self._keep_links)} links "
            f"(arm_only={bool(arm_only)}): {sorted(self._keep_links)}"
        )

        # Cache: geometry node name -> owning URDF link (last link on path).
        self._geom_link = self._build_geom_link_map()
        self._robot_geom_nodes = [
            g
            for g, link in self._geom_link.items()
            if link in self._keep_links and g in self._urdf.scene.geometry
        ]

    def _build_geom_link_map(self) -> dict:
        scene = self._urdf.scene
        G = scene.graph
        base = getattr(G, "base_frame", None) or "world"
        nx_g = G.to_networkx()
        link_names = set(self._urdf.link_map.keys())
        out = {}
        for geom_node in list(G.nodes_geometry):
            try:
                path = nx.shortest_path(nx_g, base, geom_node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            linkish = [p for p in path if p in link_names]
            if not linkish:
                continue
            out[geom_node] = linkish[-1]
        return out

    def mesh_at_q(self, q: np.ndarray) -> trimesh.Trimesh:
        """Return a gray arm+hand mesh in the URDF base / world frame."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != self.num_joints:
            raise ValueError(
                f"q must have {self.num_joints} joints, got {q.size}"
            )
        self._urdf.update_cfg(q)
        scene = self._urdf.scene

        parts = []
        for geom_node in self._robot_geom_nodes:
            geom = scene.geometry.get(geom_node)
            if geom is None:
                continue
            # Transform geometry into the scene base frame.
            T, _ = scene.graph.get(geom_node)
            mesh = geom.copy()
            mesh.apply_transform(T)
            parts.append(mesh)

        if not parts:
            raise RuntimeError(
                "No arm/hand meshes found to render; check include_links filter."
            )

        mesh = trimesh.util.concatenate(parts)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh = mesh.copy()

        # Optional topology-preserving decimation (never random face drops —
        # those punch holes and make silhouettes look like dirty wireframes).
        if self._max_faces > 0 and len(mesh.faces) > self._max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=self._max_faces)
            except Exception:
                # Fall back to full mesh if open3d / fast_simplification unavailable.
                pass
        mesh.remove_unreferenced_vertices()
        mesh.visual.vertex_colors = np.tile(
            self._gray_rgb.reshape(1, 3), (len(mesh.vertices), 1)
        )
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        return mesh
