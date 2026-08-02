"""Viser mesh overlay driven by a NamedVecList 12-vec bb2world pose."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from data_type.basic_types.NamedVecListData import NamedVecListData


class MeshVisualizer:
    def __init__(
        self,
        viser_server,
        lcm_instance,
        pose_data_queue,
        mesh_pose_data_channel,
        mesh_path,
        mesh_name="predefined_obj",
        scene_name="/mesh",
        opacity: float = 1.0,
    ):
        self._lcm_instance = lcm_instance
        self._viser_server = viser_server
        self._pose_data_queue = pose_data_queue
        self._pose_data_channel = mesh_pose_data_channel
        self._mesh_path = mesh_path
        self._mesh_name = mesh_name
        self._opacity = float(opacity)

        trimesh_mesh = trimesh.load(self._mesh_path, force="mesh")
        if isinstance(trimesh_mesh, trimesh.Scene):
            trimesh_mesh = trimesh_mesh.dump(concatenate=True)
        self._cad_to_bb, _ = trimesh.bounds.oriented_bounds(trimesh_mesh)

        if self._opacity < 1.0:
            # Make a translucent copy for GT ghost overlays.
            trimesh_mesh = trimesh_mesh.copy()
            if hasattr(trimesh_mesh.visual, "face_colors"):
                colors = np.asarray(trimesh_mesh.visual.face_colors)
                if colors.ndim == 2 and colors.shape[1] == 4:
                    colors = colors.copy()
                    colors[:, 3] = int(255 * self._opacity)
                    trimesh_mesh.visual.face_colors = colors

        self._mesh = self._viser_server.scene.add_mesh_trimesh(
            name=scene_name,
            mesh=trimesh_mesh,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self._pose = np.eye(4)
        self._set_subscribers(self._pose_data_channel, self._pose_data_queue)

    def _set_subscribers(self, pose_data_channel, pose_data_queue):
        pose_data_sub = NamedVecListSubscriber(
            self._lcm_instance,
            pose_data_queue,
        )
        pose_data_sub.subscribe(pose_data_channel)

    def _check_and_get_data_from_que(self):
        if self._pose_data_queue.empty():
            return
        while not self._pose_data_queue.empty():
            pose_data = self._pose_data_queue.get()
            if not isinstance(pose_data, NamedVecListData):
                continue
            _, name_list, vec_list = pose_data.get_data()
            if self._mesh_name not in name_list:
                continue
            vec = vec_list[name_list.index(self._mesh_name)]
            self._pose = np.eye(4)
            self._pose[:3, :3] = vec[3:12].reshape(3, 3)
            self._pose[:3, 3] = vec[:3]

    def update(self):
        self._check_and_get_data_from_que()
        cad_to_base = self._pose @ self._cad_to_bb
        self._mesh.position = cad_to_base[:3, 3]
        self._mesh.wxyz = R.from_matrix(cad_to_base[:3, :3]).as_quat(scalar_first=True)
