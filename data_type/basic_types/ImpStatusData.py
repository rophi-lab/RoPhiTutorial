"""Impedance controller status for Viser (joint vs SE(3) task)."""

from __future__ import annotations

import numpy as np

from data_type.BaseData import BaseData

MODE_JOINT = 0
MODE_TASK = 1
MODE_NAME = {MODE_JOINT: "joint", MODE_TASK: "task"}


class ImpStatusData(BaseData):
    def __init__(self, num_joints: int, name: str = "imp_status"):
        super().__init__(name)
        self.num_joints = int(num_joints)
        self.num_arm = 7
        self.mode = MODE_JOINT
        self.q = np.zeros(self.num_joints, dtype=np.float64)
        self.q_nom = np.zeros(self.num_joints, dtype=np.float64)
        self.p = np.zeros(3, dtype=np.float64)
        self.p_nom = np.zeros(3, dtype=np.float64)
        self.quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.quat_nom_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.xi = np.zeros(6, dtype=np.float64)
        self.kq_scale = 1.0
        self.dq_scale = 1.0
        self.kt_trans = 0.0
        self.kt_rot = 0.0
        self.dt_trans = 0.0
        self.dt_rot = 0.0

    def set_data(
        self,
        t: float,
        mode: int,
        q,
        q_nom,
        p,
        p_nom,
        quat_wxyz,
        quat_nom_wxyz,
        xi,
        *,
        num_arm: int = 7,
        kq_scale: float = 1.0,
        dq_scale: float = 1.0,
        kt_trans: float = 0.0,
        kt_rot: float = 0.0,
        dt_trans: float = 0.0,
        dt_rot: float = 0.0,
    ) -> None:
        self.timestamp = float(t)
        self.mode = int(mode)
        self.num_arm = int(num_arm)
        self.q = np.asarray(q, dtype=np.float64).reshape(self.num_joints).copy()
        self.q_nom = np.asarray(q_nom, dtype=np.float64).reshape(self.num_joints).copy()
        self.p = np.asarray(p, dtype=np.float64).reshape(3).copy()
        self.p_nom = np.asarray(p_nom, dtype=np.float64).reshape(3).copy()
        self.quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4).copy()
        self.quat_nom_wxyz = (
            np.asarray(quat_nom_wxyz, dtype=np.float64).reshape(4).copy()
        )
        self.xi = np.asarray(xi, dtype=np.float64).reshape(6).copy()
        self.kq_scale = float(kq_scale)
        self.dq_scale = float(dq_scale)
        self.kt_trans = float(kt_trans)
        self.kt_rot = float(kt_rot)
        self.dt_trans = float(dt_trans)
        self.dt_rot = float(dt_rot)
