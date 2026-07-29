import numpy as np

from data_type.BaseData import BaseData


class JointMeasData(BaseData):
    def __init__(self, num_joints: int, name: str = "joint_meas_data"):
        """
        Initialize the JointMeasData object.
        """
        super().__init__(name)
        self.num_joints = num_joints
        self.q = np.zeros((num_joints,))
        self.qd = np.zeros((num_joints,))
        self.tau = np.zeros((num_joints,))

    def set_data(self, t: float, q: np.ndarray, qd: np.ndarray, tau: np.ndarray):
        """
        Set the joint position, velocity, and torque data.
        """
        self.timestamp = t
        self.q = q.copy()
        self.qd = qd.copy()
        self.tau = tau.copy()

    def set_q(self, q: np.ndarray):
        """
        Set the joint position data.
        """
        self.q = q.copy()

    def set_qd(self, qd: np.ndarray):
        """
        Set the joint velocity data.
        """
        self.qd = qd.copy()

    def set_tau(self, tau: np.ndarray):
        """
        Set the joint torque data.
        """
        self.tau = tau.copy()

    def get_data(self):
        """
        Get the joint position, velocity, and torque data.
        """
        return self.timestamp, self.q.copy(), self.qd.copy(), self.tau.copy()

    def get_q(self):
        """
        Get the joint position data.
        """
        return self.q.copy()

    def get_qd(self):
        """
        Get the joint velocity data.
        """
        return self.qd.copy()

    def get_tau(self):
        """
        Get the joint torque data.
        """
        return self.tau.copy()

    def get_data_from_id(self, id):
        """
        Get the joint position, velocity, and torque data by joint id.
        """

        if id >= self.num_joints:
            raise ValueError(
                f"Joint ID {id} is out of range. Max ID is {self.num_joints - 1}."
            )
        return (
            self.timestamp,
            self.q[id].copy(),
            self.qd[id].copy(),
            self.tau[id].copy(),
        )
