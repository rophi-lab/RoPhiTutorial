"""Viser SE(3) axes frame driven by a NamedVecList 12-vec bb2world pose."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from data_type.basic_types.NamedVecListData import NamedVecListData


class PoseVisualizer:
    def __init__(
        self,
        viser_server,
        lcm_instance,
        pose_data_queue,
        pose_data_channel,
        mesh_name="predefined_obj",
        scene_name="/pose_frame",
        axes_length: float = 0.1,
    ):
        self._lcm_instance = lcm_instance
        self._viser_server = viser_server
        self._pose_data_queue = pose_data_queue
        self._pose_data_channel = pose_data_channel
        self._mesh_name = mesh_name

        self._pose = np.eye(4)
        self._set_subscribers(self._pose_data_channel, self._pose_data_queue)

        self._frame = self._viser_server.scene.add_frame(
            name=scene_name,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            axes_length=axes_length,
            axes_radius=0.002,
            origin_radius=0.002,
            origin_color=(0, 0, 0),
        )

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
            if self._mesh_name in name_list:
                vec = vec_list[name_list.index(self._mesh_name)]
            else:
                vec = vec_list[0]
            self._pose = np.eye(4)
            self._pose[:3, :3] = vec[3:12].reshape(3, 3)
            self._pose[:3, 3] = vec[:3]

    def update(self):
        self._check_and_get_data_from_que()
        self._frame.position = self._pose[:3, 3]
        self._frame.wxyz = R.from_matrix(self._pose[:3, :3]).as_quat(scalar_first=True)
