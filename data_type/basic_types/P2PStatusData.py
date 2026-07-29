"""Min-jerk P2P controller status for LCM (tracking viz + gain readout)."""

from __future__ import annotations

import numpy as np

from data_type.BaseData import BaseData

MODE_HOLD = 0
MODE_TRACK = 1
MODE_GRAV = 2

MODE_NAME = {MODE_HOLD: "hold", MODE_TRACK: "track", MODE_GRAV: "grav"}


class P2PStatusData(BaseData):
    def __init__(self, num_joints: int, name: str = "p2p_status"):
        super().__init__(name)
        self.num_joints = int(num_joints)
        self.num_arm = 7
        self.mode = MODE_HOLD
        self.q = np.zeros(self.num_joints, dtype=np.float64)
        self.q_des = np.zeros(self.num_joints, dtype=np.float64)
        self.qd = np.zeros(self.num_joints, dtype=np.float64)
        self.qd_des = np.zeros(self.num_joints, dtype=np.float64)
        self.kp = np.zeros(self.num_joints, dtype=np.float64)
        self.kd = np.zeros(self.num_joints, dtype=np.float64)
        self.fjc = np.zeros(self.num_joints, dtype=np.float64)
        self.friction_phi = 0.03
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        self.fjc_scale = 1.0
        self.do_friction_comp = True
        self.do_grav_comp = True

    def set_data(
        self,
        t: float,
        mode: int,
        q,
        q_des,
        qd,
        qd_des,
        kp,
        kd,
        fjc,
        *,
        num_arm: int = 7,
        friction_phi: float = 0.03,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        fjc_scale: float = 1.0,
        do_friction_comp: bool = True,
        do_grav_comp: bool = True,
    ) -> None:
        self.timestamp = float(t)
        self.mode = int(mode)
        self.num_arm = int(num_arm)
        self.q = np.asarray(q, dtype=np.float64).reshape(self.num_joints).copy()
        self.q_des = np.asarray(q_des, dtype=np.float64).reshape(self.num_joints).copy()
        self.qd = np.asarray(qd, dtype=np.float64).reshape(self.num_joints).copy()
        self.qd_des = (
            np.asarray(qd_des, dtype=np.float64).reshape(self.num_joints).copy()
        )
        self.kp = np.asarray(kp, dtype=np.float64).reshape(self.num_joints).copy()
        self.kd = np.asarray(kd, dtype=np.float64).reshape(self.num_joints).copy()
        self.fjc = np.asarray(fjc, dtype=np.float64).reshape(self.num_joints).copy()
        self.friction_phi = float(friction_phi)
        self.kp_scale = float(kp_scale)
        self.kd_scale = float(kd_scale)
        self.fjc_scale = float(fjc_scale)
        self.do_friction_comp = bool(do_friction_comp)
        self.do_grav_comp = bool(do_grav_comp)

    def get_data(self):
        return (
            self.timestamp,
            self.mode,
            self.q.copy(),
            self.q_des.copy(),
            self.qd.copy(),
            self.qd_des.copy(),
            self.kp.copy(),
            self.kd.copy(),
            self.fjc.copy(),
        )
