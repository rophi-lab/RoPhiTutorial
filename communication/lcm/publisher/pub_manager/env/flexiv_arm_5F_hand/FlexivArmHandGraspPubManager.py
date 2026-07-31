"""Env publisher: joints + ColInfo + grasp-object SE(3) pose + physics."""

from queue import Queue

from communication.lcm.publisher.data_publisher.ColInfoPublisher import (
    ColInfoPublisher,
)
from communication.lcm.publisher.data_publisher.JointMeasPublisher import (
    JointMeasPublisher,
)
from communication.lcm.publisher.data_publisher.NamedVecListPublisher import (
    NamedVecListPublisher,
)
from communication.lcm.publisher.data_publisher.SE3PosePublisher import (
    SE3PosePublisher,
)
from communication.lcm.publisher.BasePubManager import BaseEnvPubManager


class FlexivArmHandGraspPubManager(BaseEnvPubManager):
    """Like ``flexiv_arm_hand_col_info`` plus object pose and physics channels."""

    def __init__(
        self,
        *args,
        arm_joint_meas_channel: str = "sw_flexiv_arm_joint_meas",
        hand_joint_meas_channel: str = "sw_robotis_5F_hand_joint_meas",
        static_col_info_channel: str = "sw_flexiv_arm_hand_static_col_info",
        robot_col_info_channel: str = "sw_flexiv_arm_hand_robot_col_info",
        object_pose_channel: str = "sw_grasp_object_pose",
        object_physics_channel: str = "sw_grasp_object_physics",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pub_que_dict["intr_pub_que_dict"] = {
            arm_joint_meas_channel: Queue(),
            hand_joint_meas_channel: Queue(),
            robot_col_info_channel: Queue(),
        }
        self.pub_que_dict["extr_pub_que_dict"] = {
            static_col_info_channel: Queue(),
            object_pose_channel: Queue(maxsize=1),
        }
        if object_physics_channel:
            self.pub_que_dict["extr_pub_que_dict"][object_physics_channel] = Queue(
                maxsize=1
            )

        self.publisher_dict[arm_joint_meas_channel] = JointMeasPublisher(
            self._lcm_instance,
            arm_joint_meas_channel,
            self.pub_que_dict["intr_pub_que_dict"][arm_joint_meas_channel],
        )
        self.publisher_dict[hand_joint_meas_channel] = JointMeasPublisher(
            self._lcm_instance,
            hand_joint_meas_channel,
            self.pub_que_dict["intr_pub_que_dict"][hand_joint_meas_channel],
        )
        self.publisher_dict[robot_col_info_channel] = ColInfoPublisher(
            self._lcm_instance,
            robot_col_info_channel,
            self.pub_que_dict["intr_pub_que_dict"][robot_col_info_channel],
        )
        self.publisher_dict[static_col_info_channel] = ColInfoPublisher(
            self._lcm_instance,
            static_col_info_channel,
            self.pub_que_dict["extr_pub_que_dict"][static_col_info_channel],
        )
        self.publisher_dict[object_pose_channel] = SE3PosePublisher(
            self._lcm_instance,
            object_pose_channel,
            self.pub_que_dict["extr_pub_que_dict"][object_pose_channel],
        )
        if object_physics_channel:
            self.publisher_dict[object_physics_channel] = NamedVecListPublisher(
                self._lcm_instance,
                object_physics_channel,
                self.pub_que_dict["extr_pub_que_dict"][object_physics_channel],
            )
