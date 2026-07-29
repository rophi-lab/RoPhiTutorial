from queue import Queue

from communication.lcm.publisher.BasePubManager import BaseCtrlPubManager
from communication.lcm.publisher.data_publisher.JointCtrlPublisher import (
    JointCtrlPublisher,
)
from communication.lcm.publisher.data_publisher.Pose3DPublisher import Pose3DPublisher


class JointCtrlPosePubManager(BaseCtrlPubManager):
    """Publish joint control commands and a 3D pose target (IK setpoint)."""

    def __init__(
        self,
        ctrl_channel: str = "sw_flexiv_arm_hand_joint_ctrl",
        pose_channel: str = "sw_ik_target_pose",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.publisher_dict[ctrl_channel] = JointCtrlPublisher(
            self._lcm_instance, ctrl_channel, self.ctrl_pub_que
        )

        self.pub_que_dict[pose_channel] = Queue(maxsize=1)
        self.publisher_dict[pose_channel] = Pose3DPublisher(
            self._lcm_instance,
            pose_channel,
            self.pub_que_dict[pose_channel],
        )
