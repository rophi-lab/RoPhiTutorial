"""World-frame SE(3) pose for LCM (position + quaternion wxyz)."""

from __future__ import annotations

import numpy as np

from data_type.BaseData import BaseData


class SE3PoseData(BaseData):
    def __init__(self, name: str = "se3_pose"):
        super().__init__(name)
        self.position = np.zeros(3, dtype=np.float64)
        self.quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def set_data(self, t: float, position, quat_wxyz) -> None:
        self.timestamp = float(t)
        self.position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4).copy()
        n = float(np.linalg.norm(q))
        self.quat_wxyz = q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

    def set_from_matrix(self, t: float, T: np.ndarray) -> None:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        R = T[:3, :3]
        # Same conversion as impedance controller.
        tr = float(np.trace(R))
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (
                R[1, 0] - R[0, 1]
            ) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (
                R[0, 2] + R[2, 0]
            ) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (
                R[1, 2] + R[2, 1]
            ) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (
                R[1, 2] + R[2, 1]
            ) / s, 0.25 * s
        self.set_data(t, T[:3, 3], [w, x, y, z])

    def get_data(self):
        return self.timestamp, self.position.copy(), self.quat_wxyz.copy()
