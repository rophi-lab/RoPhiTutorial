import numpy as np

from data_type.BaseData import BaseData


class JointTrajData(BaseData):
    """Discrete joint-space trajectory (t, q, qd) for LCM / visualization."""

    def __init__(self, name: str = "joint_traj_data"):
        super().__init__(name=name)
        self.num_joints = 0
        self.num_points = 0
        self.t = np.zeros(0, dtype=np.float64)
        self.q = np.zeros((0, 0), dtype=np.float64)
        self.qd = np.zeros((0, 0), dtype=np.float64)

    def set_data(
        self,
        timestamp: float,
        t: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> None:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        if q.ndim != 2 or qd.ndim != 2:
            raise ValueError(f"q/qd must be (N, n); got {q.shape}, {qd.shape}")
        if t.shape[0] != q.shape[0] or q.shape != qd.shape:
            raise ValueError(
                f"Shape mismatch: t={t.shape}, q={q.shape}, qd={qd.shape}"
            )
        self.set_time(timestamp)
        self.t = t.copy()
        self.q = q.copy()
        self.qd = qd.copy()
        self.num_points = int(q.shape[0])
        self.num_joints = int(q.shape[1])

    def get_data(self):
        return (
            self.timestamp,
            self.t.copy(),
            self.q.copy(),
            self.qd.copy(),
        )
