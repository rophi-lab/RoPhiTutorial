import copy

from queue import Queue

import numpy as np

import mujoco as mj

from robot_platform.sim.BaseSimPlatform import BaseSimPlatform
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData


class SimRobotis5FHand(BaseSimPlatform):
    """
    Robotis 5F hand platform
    """

    def __init__(
        self,
        *args,
        joint_meas_freq=1000,
        default_q_values=[0] * 20,
        joint_meas_channel="robotis_5F_hand_joint_meas",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.joint_name = [
            "finger_r_joint_1_thumb1",
            "finger_r_joint_1_thumb2",
            "finger_r_joint_1_thumb3",
            "finger_r_joint_1_thumb4",
            "finger_r_joint_2_index1",
            "finger_r_joint_2_index2",
            "finger_r_joint_2_index3",
            "finger_r_joint_2_index4",
            "finger_r_joint_3_middle1",
            "finger_r_joint_3_middle2",
            "finger_r_joint_3_middle3",
            "finger_r_joint_3_middle4",
            "finger_r_joint_4_ring1",
            "finger_r_joint_4_ring2",
            "finger_r_joint_4_ring3",
            "finger_r_joint_4_ring4",
            "finger_r_joint_5_little1",
            "finger_r_joint_5_little2",
            "finger_r_joint_5_little3",
            "finger_r_joint_5_little4",
        ]

        self.meas_data = JointMeasData(num_joints=20)

        self._default_q = np.array(default_q_values)
        self._last_ctrl_data = JointCtrlData(num_joints=20)

        self._joint_meas_channel = joint_meas_channel

        self._list_of_joint_indices = []
        self._last_pub_time = {self._joint_meas_channel: 0.0}
        self._pub_dt = {self._joint_meas_channel: 1.0 / joint_meas_freq}

    def set_mj_data_name_idx(self, mj_data: mj.MjData):
        """
        Set the joint names and indices in the Mujoco data object.
        @param[in] mj_data: Mujoco data object.
        """
        self._list_of_joint_indices = []
        for joint_name in self.joint_name:
            joint_index = mj.mj_name2id(
                mj_data.model, mj.mjtObj.mjOBJ_JOINT, joint_name
            )
            self._list_of_joint_indices.append(joint_index)

    # def get_model_and_anchor_name(self, xml_path: str):
    #     return mj.MjSpec.from_file(xml_path), "sh_yaw_link"

    def apply_sim_control(self, ctrl_data: JointCtrlData, mj_data: mj.MjData):
        """
        Apply control data to the platform.
        @param[in] ctrl_data: Control data to apply.
        @param[out] mj_data: Mujoco data object.
        """
        if ctrl_data is None:
            ctrl_data = self._last_ctrl_data

        t, q, qd, tau = self.meas_data.get_data()
        t, q_des, qd_des, tau_ff, kp, kd = ctrl_data.get_data()
        # update the last control data
        self._last_ctrl_data.set_data(t, q_des, qd_des, tau_ff, kp, kd)
        tau_command = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
        mj_data.ctrl[self._list_of_joint_indices] = tau_command

    def add_friction_and_inertial_correction_to_sim(
        self, mj_data: mj.MjData, qdd_estimated: np.ndarray
    ):
        """
        Add friction and inertial correction to the mj_data_ctrl.
        @param[in/out] mj_data: Mujoco data object.
        @param[in] qdd_estimated: Estimated qdd.
        """
        pass

    def sync_intr_data_from_sim(self, mj_data: mj.MjData, intr_pub_que_dict: dict):
        """
        Synchronize data from the simulation and update self._measure_data.
        @param[in/out] mj_data: Mujoco data object.
        @param[out] intr_pub_que_dict: Dictionary of internal publisher queues.
        """

        t = mj_data.time
        # to enforce the publication rate
        if (
            t - self._last_pub_time[self._joint_meas_channel]
            > self._pub_dt[self._joint_meas_channel]
        ):
            q = mj_data.qpos[self._list_of_joint_indices]
            qd = mj_data.qvel[self._list_of_joint_indices]
            tau = mj_data.qfrc_actuator[self._list_of_joint_indices]
            self.meas_data.set_data(t, q, qd, tau)

            intr_pub_que_dict[self._joint_meas_channel].put(
                copy.deepcopy(self.meas_data)
            )
            self._last_pub_time[self._joint_meas_channel] = t

    def reset(self, mj_data: mj.MjData):
        self._last_ctrl_data.set_q_des(self._default_q)
        self.meas_data.set_q(self._default_q)

        mj_data.qpos[self._list_of_joint_indices] = self._default_q
        mj_data.qvel[self._list_of_joint_indices] = np.zeros((20,))
        mj_data.qfrc_actuator[self._list_of_joint_indices] = np.zeros((20,))
        mj_data.ctrl[self._list_of_joint_indices] = np.zeros((20,))

        self._last_pub_time[self._joint_meas_channel] = mj_data.time

    def get_col_info_data(self):
        """
        Get the collision information data.
        @return: Collision information data.
        """
        return [
            {
                "name": "palm",
                "geom_type": "box",
                "size": np.array([0.075, 0.09, 0.15]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.05],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            # Thumb
            {
                "name": "finger_r_link_1_thumb3",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.034]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_1_thumb4",
                "geom_type": "cylinder",
                "size": np.array([0.015, 0.036]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            # Index
            {
                "name": "finger_r_link_2_index2",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.03]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_2_index3",
                "geom_type": "cylinder",
                "size": np.array([0.015, 0.034]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_2_index4",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.036]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            # Middle
            {
                "name": "finger_r_link_3_middle2",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.03]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_3_middle3",
                "geom_type": "cylinder",
                "size": np.array([0.015, 0.034]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_3_middle4",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.036]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            # Ring
            {
                "name": "finger_r_link_4_ring2",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.03]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_4_ring3",
                "geom_type": "cylinder",
                "size": np.array([0.015, 0.034]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_4_ring4",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.036]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            # Little
            {
                "name": "finger_r_link_5_little2",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.03]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_5_little3",
                "geom_type": "cylinder",
                "size": np.array([0.015, 0.034]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            {
                "name": "finger_r_link_5_little4",
                "geom_type": "cylinder",
                "size": np.array([0.013, 0.036]),
                "offset": np.array(
                    [
                        [1.0, 0.0, 0.0, -0.001],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.015],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
        ]
