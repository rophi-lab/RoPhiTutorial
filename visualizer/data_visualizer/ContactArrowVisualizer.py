"""Live contact arrow visualizer for Viser.

Subscribes to ``NamedVecListData`` rows of ``[px, py, pz, vx, vy, vz]`` (world
frame) and draws line segments + contact-point spheres. Used for both contact
forces and contact normals (unit direction, fixed visual scale).

A single row named ``"empty"`` clears previously rendered arrows.
"""

from __future__ import annotations

import numpy as np

from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from data_type.basic_types.NamedVecListData import NamedVecListData


class ContactArrowVisualizer:
    def __init__(
        self,
        viser_server,
        lcm_instance,
        data_queue,
        channel: str,
        scene_name: str,
        scale: float = 0.05,
        color=(1.0, 0.0, 0.8),
        line_width: float = 2.5,
        point_radius: float = 0.003,
        normalize_direction: bool = False,
        *args,
        **kwargs,
    ):
        self._viser_server = viser_server
        self._queue = data_queue
        self._channel = channel
        self._scene_name = scene_name
        self._scale = float(scale)
        self._color = tuple(color)
        self._line_width = float(line_width)
        self._point_radius = float(point_radius)
        self._normalize = bool(normalize_direction)

        self._latest = None  # (P, 6) or None
        self._line_handle = None
        self._point_handles: list = []
        self._dirty = True

        NamedVecListSubscriber(lcm_instance, self._queue).subscribe(channel)

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            self._dirty = True
            data = self._queue.get()
            if not isinstance(data, NamedVecListData):
                continue
            _, name_list, vec_list = data.get_data()
            vec_list = np.asarray(vec_list, dtype=np.float64)
            if (
                vec_list.shape[0] == 1
                and len(name_list) == 1
                and str(name_list[0]) == "empty"
            ):
                self._latest = None
            elif vec_list.ndim == 2 and vec_list.shape[1] >= 6:
                self._latest = vec_list[:, :6]
            else:
                self._latest = None

    def _clear_handles(self) -> None:
        if self._line_handle is not None:
            self._line_handle.remove()
            self._line_handle = None
        for h in self._point_handles:
            h.remove()
        self._point_handles.clear()

    def update(self) -> None:
        self._drain_queue()
        if not self._dirty:
            return
        self._dirty = False
        self._clear_handles()

        if self._latest is None or self._latest.shape[0] == 0:
            return

        points = self._latest[:, :3]
        dirs = self._latest[:, 3:6].copy()
        if self._normalize:
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs / np.maximum(norms, 1e-8)

        P = points.shape[0]
        segs = np.zeros((P, 2, 3), dtype=np.float64)
        segs[:, 0, :] = points
        segs[:, 1, :] = points + self._scale * dirs

        self._line_handle = self._viser_server.scene.add_line_segments(
            f"/{self._scene_name}/{self._channel}",
            segs,
            colors=self._color,
            line_width=self._line_width,
        )
        for i in range(P):
            h = self._viser_server.scene.add_icosphere(
                f"/{self._scene_name}/{self._channel}/pt_{i}",
                radius=self._point_radius,
                color=self._color,
                position=tuple(points[i]),
            )
            self._point_handles.append(h)
