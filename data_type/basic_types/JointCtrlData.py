import numpy as np

from data_type.BaseData import BaseData


class JointCtrlData(BaseData):
    def __init__(self, num_joints: int, name: str = "joint_ctrl_data"):
        super().__init__(name=name)
        self.num_joints = num_joints
        self.q_des = np.zeros((num_joints,))
        self.qd_des = np.zeros((num_joints,))
        self.tau_ff = np.zeros((num_joints,))
        self.kp = np.zeros((num_joints,))
        self.kd = np.zeros((num_joints,))
        self.valid = True

    def set_data(
        self,
        t: float,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ):
        """
        Set the data for the JointCtrlData object.
        """
        self.set_time(t)
        self.set_q_des(q_des.copy())
        self.set_qd_des(qd_des.copy())
        self.set_tau_ff(tau_ff.copy())
        self.set_kp(kp.copy())
        self.set_kd(kd.copy())

    def set_q_des(self, q_des: np.ndarray):
        """
        Set the desired joint position.
        """
        self.q_des = q_des.copy()

    def set_qd_des(self, qd_des: np.ndarray):
        """
        Set the desired joint velocity.
        """
        self.qd_des = qd_des.copy()

    def set_tau_ff(self, tau_ff: np.ndarray):
        """
        Set the feedforward torque.
        """
        self.tau_ff = tau_ff.copy()

    def set_kp(self, kp: np.ndarray):
        """
        Set the proportional gain.
        """
        self.kp = kp.copy()

    def set_kd(self, kd: np.ndarray):
        """
        Set the derivative gain.
        """
        self.kd = kd.copy()

    def set_valid(self, flag: bool):
        """
        Set the valid flag.
        """
        self.valid = flag

    def get_data(self):
        return (
            self.timestamp,
            self.q_des.copy(),
            self.qd_des.copy(),
            self.tau_ff.copy(),
            self.kp.copy(),
            self.kd.copy(),
        )

    def get_q_des(self):
        return self.q_des.copy()

    def get_qd_des(self):
        return self.qd_des.copy()

    def get_tau_ff(self):
        return self.tau_ff.copy()

    def get_kp(self):
        return self.kp.copy()

    def get_kd(self):
        return self.kd.copy()

    def get_valid(self):
        return self.valid

    def get_data_from_id(self, id):
        """
        Get the data from the data object by name.
        """

        if id >= self.num_joints:
            raise ValueError(
                f"Joint id {id} is out of range. Max id is {self.num_joints - 1}."
            )

        return (
            self.timestamp,
            self.q_des[id].copy(),
            self.qd_des[id].copy(),
            self.tau_ff[id].copy(),
            self.kp[id].copy(),
            self.kd[id].copy(),
        )
