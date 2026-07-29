import numpy as np

from data_type.BaseData import BaseData


class Pose3DData(BaseData):
    """World-frame 3D target pose + tip velocity for LCM publish/subscribe.

    ``position`` — desired task-point target (sphere in Viser).
    ``velocity`` — desired Cartesian tip velocity ``v*`` (arrow in Viser).
    ``tip`` — current task-point / fingertip (arrow origin).
    """

    def __init__(self, name: str = "pose3d"):
        super().__init__(name)
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.tip = np.zeros(3, dtype=np.float64)

    def set_data(self, t: float, position, velocity=None, tip=None):
        self.timestamp = float(t)
        self.position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.velocity = (
            np.zeros(3, dtype=np.float64)
            if velocity is None
            else np.asarray(velocity, dtype=np.float64).reshape(3).copy()
        )
        self.tip = (
            self.position.copy()
            if tip is None
            else np.asarray(tip, dtype=np.float64).reshape(3).copy()
        )

    def get_data(self):
        return (
            self.timestamp,
            self.position.copy(),
            self.velocity.copy(),
            self.tip.copy(),
        )
