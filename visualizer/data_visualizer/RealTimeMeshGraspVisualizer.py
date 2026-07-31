"""
Real-time visualizer for MuJoCo grasp evaluation.

Displays:
  - Object mesh positioned by real-time bb2world pose (NamedVecListData)
  - Hand URDF driven by real-time joint measurements (JointMeasData)
  - Contact force arrows at contact points (NamedVecListData, optional)

Unlike MeshGraspVisualizer (which browses pre-computed grasps from files),
this class shows the live simulation state.
"""

import os
import numpy as np

from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)
from data_type.basic_types.NamedVecListData import NamedVecListData
from data_type.basic_types.JointMeasData import JointMeasData

import trimesh
from scipy.spatial.transform import Rotation as R

from yourdfpy import URDF
from viser.extras import ViserUrdf
import time


class RealTimeMeshGraspVisualizer:
    def __init__(
        self,
        viser_server,
        lcm_instance,
        # Object pose
        obj_pose_data_queue,
        obj_pose_data_channel,
        # Hand joints
        hand_joint_data_queue,
        hand_joint_data_channel,
        num_hand_joints=20,
        # Contact forces (optional)
        contact_forces_data_queue=None,
        contact_forces_data_channel=None,
        # Mesh config
        mesh_path="",
        obj_name="",
        mesh_name="predefined_obj",
        # Hand config
        hand_urdf_path="",
        hand_mesh_path="",
        hand_alpha=0.5,
        # Force vis config
        force_scale=0.1,
        **kwargs,
    ):
        self._viser_server = viser_server
        self._lcm_instance = lcm_instance

        # Queues
        self._obj_pose_queue = obj_pose_data_queue
        self._obj_pose_channel = obj_pose_data_channel
        self._hand_joint_queue = hand_joint_data_queue
        self._hand_joint_channel = hand_joint_data_channel
        self._num_hand_joints = num_hand_joints
        self._contact_forces_queue = contact_forces_data_queue
        self._contact_forces_channel = contact_forces_data_channel
        self._force_scale = force_scale

        self._mesh_name = mesh_name
        self._mesh_path = mesh_path

        # ---------- Object mesh ----------
        trimesh_mesh = trimesh.load(self._mesh_path)
        if isinstance(trimesh_mesh, trimesh.Scene):
            trimesh_mesh = trimesh_mesh.dump(concatenate=True)
        # Remove degenerate faces to avoid NaN normals crashing viser
        trimesh_mesh.update_faces(trimesh_mesh.nondegenerate_faces())
        trimesh_mesh.fix_normals()
        self._cad_to_bb, _ = trimesh.bounds.oriented_bounds(trimesh_mesh)

        self._mesh_handle = self._viser_server.scene.add_mesh_trimesh(
            name="/mesh",
            mesh=trimesh_mesh,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self._pose = np.eye(4)  # bb2world

        # # ---------- Ghost mesh (perceived object pose with error) ----------
        ghost_mesh = trimesh_mesh.copy()
        if not isinstance(ghost_mesh.visual, trimesh.visual.TextureVisuals):
            ghost_mesh.visual = ghost_mesh.visual.to_texture()
        ghost_mesh.visual.material = ghost_mesh.visual.material.to_pbr()
        ghost_mesh.visual.material.alphaMode = "BLEND"
        ghost_mesh.visual.material.baseColorFactor = [1.0, 0.5, 0.5, 0.5]  # Semi-transparent red
        self._ghost_mesh_handle = self._viser_server.scene.add_mesh_trimesh(
            name="/ghost_mesh",
            mesh=ghost_mesh,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self._ghost_mesh_handle.visible = False
        self._ghost_pose = None

        # ---------- Hand URDF ----------
        load_meshes = True
        urdf = URDF.load(
            hand_urdf_path,
            mesh_dir=hand_mesh_path,
            load_meshes=load_meshes,
            build_scene_graph=load_meshes,
            load_collision_meshes=False,
            build_collision_scene_graph=False,
        )
        # Apply transparency
        # for geometry in urdf.scene.geometry.values():
        #     if isinstance(geometry, trimesh.Trimesh):
        #         geometry.visual = geometry.visual.to_texture()
        #         geometry.visual.material = geometry.visual.material.to_pbr()
        #         geometry.visual.material.alphaMode = "BLEND"
        #         geometry.visual.material.baseColorFactor = [
        #             0.5, 0.5, 0.5, hand_alpha,
        #         ]

        self._hand = ViserUrdf(
            self._viser_server,
            root_node_name="/hand",
            urdf_or_path=urdf,
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._q = np.zeros(self._num_hand_joints)

        # ---------- Force arrows (line segments) ----------
        self._force_line_handle = None
        self._contact_point_handles = []
        self._normal_line_handle = None
        self._contact_forces_data = None  # (P, 6): [px,py,pz,fx,fy,fz] in world frame
        self._contact_normals_data = None  # (P, 6): [px,py,pz,nx,ny,nz]

        # ---------- Perturbation arrow (pre-allocated, separate style) ----------
        self._perturbation_data = None  # (6,): [px,py,pz,fx,fy,fz] or None
        self._perturb_point_handle = self._viser_server.scene.add_icosphere(
            "/perturbation/point",
            radius=0.005,
            color=(0.0, 0.9, 1.0),  # Cyan
        )
        self._perturb_point_handle.visible = False
        self._perturb_line_handle = self._viser_server.scene.add_line_segments(
            "/perturbation/line",
            np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            colors=(0.0, 0.9, 1.0),  # Cyan
            line_width=4.0,
        )
        self._perturb_line_handle.visible = False

        # ---------- Subscribers ----------
        obj_pose_sub = NamedVecListSubscriber(self._lcm_instance, self._obj_pose_queue)
        obj_pose_sub.subscribe(self._obj_pose_channel)

        hand_joint_sub = JointMeasSubscriber(
            self._lcm_instance, self._hand_joint_queue, self._num_hand_joints,
        )
        hand_joint_sub.subscribe(self._hand_joint_channel)

        if self._contact_forces_queue is not None and self._contact_forces_channel is not None:
            forces_sub = NamedVecListSubscriber(self._lcm_instance, self._contact_forces_queue)
            forces_sub.subscribe(self._contact_forces_channel)

    # ------------------------------------------------------------------
    def _drain_queues(self):
        # Object pose (bb2world)
        while not self._obj_pose_queue.empty():
            data = self._obj_pose_queue.get()
            if isinstance(data, NamedVecListData):
                _, name_list, vec_list = data.get_data()
                if self._mesh_name in name_list:
                    vec = vec_list[name_list.index(self._mesh_name)]
                    self._pose = np.eye(4)
                    self._pose[:3, :3] = vec[3:12].reshape(3, 3)
                    self._pose[:3, 3] = vec[:3]
                ghost_name = self._mesh_name + "_ghost"
                if ghost_name in name_list:
                    vec = vec_list[name_list.index(ghost_name)]
                    self._ghost_pose = np.eye(4)
                    self._ghost_pose[:3, :3] = vec[3:12].reshape(3, 3)
                    self._ghost_pose[:3, 3] = vec[:3]
                else:
                    self._ghost_pose = None

        # Hand joints
        while not self._hand_joint_queue.empty():
            data = self._hand_joint_queue.get()
            if isinstance(data, JointMeasData):
                _, q, _, _ = data.get_data()
                self._q = q

        # Contact forces + normals + perturbation (separated by name)
        if self._contact_forces_queue is not None:
            while not self._contact_forces_queue.empty():
                data = self._contact_forces_queue.get()
                if isinstance(data, NamedVecListData):
                    _, name_list, vec_list = data.get_data()
                    vec_list = np.asarray(vec_list)
                    if len(name_list) == 1 and str(name_list[0]) == "empty":
                        self._contact_forces_data = None
                        self._contact_normals_data = None
                        self._perturbation_data = None
                        continue

                    force_mask = [
                        (n != "perturbation")
                        and (n != "empty")
                        and (n != "net")
                        and (not str(n).startswith("n_"))
                        for n in name_list
                    ]
                    normal_mask = [str(n).startswith("n_") for n in name_list]
                    perturb_mask = [n == "perturbation" for n in name_list]

                    if any(force_mask):
                        self._contact_forces_data = vec_list[force_mask]
                    # Do not clear contacts on perturbation-only messages.
                    # `"empty"` (handled above) is the explicit clear signal.
                    if any(normal_mask):
                        self._contact_normals_data = vec_list[normal_mask]
                    if any(perturb_mask):
                        self._perturbation_data = vec_list[perturb_mask][0]
                    else:
                        # Only clear perturbation when an explicit empty/clear
                        # message arrives (handled above) or contact packet
                        # omits it while including force rows.
                        if any(force_mask):
                            self._perturbation_data = None

    # ------------------------------------------------------------------
    def update(self):
        self._drain_queues()

        # --- Object mesh ---
        cad_to_world = self._pose @ self._cad_to_bb
        self._mesh_handle.position = cad_to_world[:3, 3]
        self._mesh_handle.wxyz = R.from_matrix(cad_to_world[:3, :3]).as_quat(
            scalar_first=True
        )

        # --- Ghost mesh (perceived pose) ---
        if self._ghost_pose is not None:
            ghost_cad_to_world = self._ghost_pose @ self._cad_to_bb
            self._ghost_mesh_handle.position = ghost_cad_to_world[:3, 3]
            self._ghost_mesh_handle.wxyz = R.from_matrix(
                ghost_cad_to_world[:3, :3]
            ).as_quat(scalar_first=True)
            self._ghost_mesh_handle.visible = True
        else:
            self._ghost_mesh_handle.visible = False

        # --- Hand ---
        self._hand.update_cfg(self._q)

        # --- Force arrows (world frame, attached to hand) ---
        self._update_force_arrows()

        # --- Perturbation arrow (world frame, at object center) ---
        self._update_perturbation_arrow()

    def _update_force_arrows(self):
        """Update QP contact points (spheres) + force arrows + normals in world."""
        if self._force_line_handle is not None:
            self._force_line_handle.remove()
            self._force_line_handle = None
        if self._normal_line_handle is not None:
            self._normal_line_handle.remove()
            self._normal_line_handle = None
        for h in self._contact_point_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._contact_point_handles = []

        if self._contact_forces_data is None or len(self._contact_forces_data) == 0:
            return

        points_world = self._contact_forces_data[:, :3]
        forces_world = self._contact_forces_data[:, 3:6]
        P = points_world.shape[0]

        # Contact points used in the QP
        for i in range(P):
            h = self._viser_server.scene.add_icosphere(
                f"/contact_forces/pt_{i}",
                radius=0.004,
                color=(1.0, 0.0, 0.0),
                position=points_world[i],
            )
            self._contact_point_handles.append(h)

        line_segments = np.zeros((P, 2, 3))
        line_segments[:, 0, :] = points_world
        line_segments[:, 1, :] = points_world + self._force_scale * forces_world
        self._force_line_handle = self._viser_server.scene.add_line_segments(
            "/contact_forces/arrows",
            line_segments,
            colors=(1.0, 0.0, 0.8),
            line_width=2.5,
        )

        if self._contact_normals_data is not None and len(self._contact_normals_data) > 0:
            n_pts = self._contact_normals_data[:, :3]
            n_dir = self._contact_normals_data[:, 3:6]
            N = n_pts.shape[0]
            n_seg = np.zeros((N, 2, 3))
            n_seg[:, 0, :] = n_pts
            n_seg[:, 1, :] = n_pts + 0.015 * n_dir
            self._normal_line_handle = self._viser_server.scene.add_line_segments(
                "/contact_forces/normals",
                n_seg,
                colors=(0.0, 1.0, 0.0),
                line_width=2.0,
            )

    def _update_perturbation_arrow(self):
        """Update perturbation force arrow (cyan, thicker, pre-allocated handles)."""
        if self._perturbation_data is None:
            self._perturb_point_handle.visible = False
            self._perturb_line_handle.visible = False
            return

        start = self._perturbation_data[:3]
        force = self._perturbation_data[3:6]
        end = start + self._force_scale * force

        # Update point
        self._perturb_point_handle.position = start
        self._perturb_point_handle.visible = True

        # Update line (re-add with same name — replaces in-place)
        self._perturb_line_handle = self._viser_server.scene.add_line_segments(
            "/perturbation/line",
            np.array([[start, end]]),
            colors=(0.0, 0.9, 1.0),  # Cyan
            line_width=4.0,
        )
        self._perturb_line_handle.visible = True
