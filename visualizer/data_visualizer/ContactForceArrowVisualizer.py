"""Live contact force arrow visualizer.

Subscribes to a ``NamedVecListData`` channel whose rows encode one contact
force arrow each as ``[px, py, pz, fx, fy, fz]`` in world frame, and renders
them as line segments in viser.

Mirrors the rendering style used by
``visualizer.data_visualizer.RealTimeMeshGraspVisualizer._update_force_arrows``
so the realtime FF reactive-grasp controller's contact wrenches look the same
as the mj_eval ones.

Special row name ``"empty"`` (with zero data) is treated as "no contacts" and
clears any previously rendered arrows; this lets the controller deterministically
clean up stale visualization when force closure drops to zero contacts.
"""

import numpy as np

from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from data_type.basic_types.NamedVecListData import NamedVecListData


class ContactForceArrowVisualizer:
    def __init__(
        self,
        viser_server,
        lcm_instance,
        contact_force_arrows_data_queue,
        contact_force_arrows_data_channel,
        force_scale=0.05,
        color=(1.0, 0.0, 0.8),  # Magenta, matches MuJoCo viewer
        line_width=2.5,
        *args,
        **kwargs,
    ):
        self._viser_server = viser_server
        self._lcm_instance = lcm_instance
        self._queue = contact_force_arrows_data_queue
        self._channel = contact_force_arrows_data_channel
        self._force_scale = float(force_scale)
        self._color = tuple(color)
        self._line_width = float(line_width)

        self._latest = None  # (P, 6) in world frame, or None
        self._line_handle = None
        self._point_handles = []

        self._set_subscribers()

    def _set_subscribers(self):
        sub = NamedVecListSubscriber(self._lcm_instance, self._queue)
        sub.subscribe(self._channel)

    def _drain_queue(self):
        while not self._queue.empty():
            data = self._queue.get()
            if not isinstance(data, NamedVecListData):
                raise ValueError(
                    f"Expected NamedVecListData on contact_force_arrows channel, "
                    f"got {type(data)}"
                )
            _, name_list, vec_list = data.get_data()
            vec_list = np.asarray(vec_list)
            # Sentinel: a single row named "empty" means "no contacts now".
            if (
                vec_list.shape[0] == 1
                and len(name_list) == 1
                and name_list[0] == "empty"
            ):
                self._latest = None
            else:
                self._latest = vec_list  # (P, 6)

    def update(self):
        self._drain_queue()

        # Always remove the previous handles so transient contacts don't leave
        # ghost arrows behind.
        if self._line_handle is not None:
            self._line_handle.remove()
            self._line_handle = None
        for h in self._point_handles:
            h.remove()
        self._point_handles.clear()

        if self._latest is None or self._latest.shape[0] == 0:
            return

        points_world = self._latest[:, :3]
        forces_world = self._latest[:, 3:6]
        P = points_world.shape[0]

        line_segments = np.zeros((P, 2, 3))
        line_segments[:, 0, :] = points_world
        line_segments[:, 1, :] = points_world + self._force_scale * forces_world

        self._line_handle = self._viser_server.scene.add_line_segments(
            f"/contact_force_arrows/{self._channel}",
            line_segments,
            colors=self._color,
            line_width=self._line_width,
        )

        for i in range(P):
            h = self._viser_server.scene.add_icosphere(
                f"/contact_force_arrows/{self._channel}/pt_{i}",
                radius=0.003,
                color=self._color,
                position=points_world[i],
            )
            self._point_handles.append(h)
