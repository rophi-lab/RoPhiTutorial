"""Gravity-compensation controller (no collision avoidance)."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from omegaconf import DictConfig

from controller.BaseController import BaseController
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData


class GravCompController(BaseController):
    """Hold the robot with Pinocchio gravity feedforward.

    Publishes::

        tau_ff = g(q) + arm_damping * (-qd_arm)
        kp = 0, kd = 0
        q_des = q, qd_des = 0

    so the plant (sim or hardware) only needs to apply ``tau_ff``.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._dict_joints = config["dict_joints"]
        self._num_joints = int(config["num_joints"])
        self._num_arm_joints = int(config.get("num_arm_joints", 7))

        self._pin_model = pin.buildModelFromUrdf(config["urdf_path"])
        self._pin_data = self._pin_model.createData()

        self._q = np.zeros(self._num_joints, dtype=np.float64)
        self._qd = np.zeros(self._num_joints, dtype=np.float64)

        # Light viscous damping on the arm only (stabilizes free-float gravcomp).
        self._arm_damping = float(config.get("arm_damping", 0.1))

        default_q = config.get("default_q", None)
        if default_q is not None:
            self._q[:] = np.asarray(default_q, dtype=np.float64)

    def initialize(self) -> None:
        """No async setup required for pure gravity compensation."""
        print("[GravComp] Initialized.")

    def _check_and_get_data_from_que(self) -> None:
        """Pull the latest joint measurements into ``_q`` / ``_qd``."""
        for joint_cfg in self._dict_joints.values():
            channel = joint_cfg["joint_meas_channel"]
            idx = joint_cfg["list_joint_idx"]
            if self.intr_sub_que_dict[channel].empty():
                continue

            meas = self.intr_sub_que_dict[channel].get()
            if not isinstance(meas, JointMeasData):
                raise ValueError(
                    f"Expected JointMeasData on '{channel}', got {type(meas)}"
                )
            _, q, qd, _ = meas.get_data()
            self._q[idx] = q
            self._qd[idx] = qd

    def _update(self) -> None:
        """Compute gravity torque and publish a zero-gain JointCtrl command."""
        tau_ff = pin.computeGeneralizedGravity(self._pin_model, self._pin_data, self._q)
        arm = slice(0, self._num_arm_joints)
        tau_ff[arm] = tau_ff[arm] - self._arm_damping * self._qd[arm]

        cmd = JointCtrlData(num_joints=self._num_joints)
        cmd.set_data(
            self._cur_time,
            self._q.copy(),
            np.zeros(self._num_joints),
            tau_ff,
            np.zeros(self._num_joints),
            np.zeros(self._num_joints),
        )
        self.ctrl_pub_que.put(cmd)
