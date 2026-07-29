from queue import Queue

from communication.lcm.publisher.BasePubManager import BaseCtrlPubManager
from communication.lcm.publisher.data_publisher.JointCtrlPublisher import (
    JointCtrlPublisher,
)
from communication.lcm.publisher.data_publisher.JointMeasPublisher import (
    JointMeasPublisher,
)
from communication.lcm.publisher.data_publisher.JointTrajPublisher import (
    JointTrajPublisher,
)
from communication.lcm.publisher.data_publisher.P2PStatusPublisher import (
    P2PStatusPublisher,
)
from communication.lcm.publisher.data_publisher.SE3PosePublisher import (
    SE3PosePublisher,
)


class JointCtrlTrajPubManager(BaseCtrlPubManager):
    """Publish joint ctrl, target config, traj, P2P status, optional SE(3) pose."""

    def __init__(
        self,
        ctrl_channel: str = "sw_flexiv_arm_hand_joint_ctrl",
        target_joint_channel: str = "sw_p2p_target_joint",
        traj_channel: str = "sw_p2p_joint_traj",
        status_channel: str = "sw_p2p_status",
        palm_pose_channel: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.publisher_dict[ctrl_channel] = JointCtrlPublisher(
            self._lcm_instance, ctrl_channel, self.ctrl_pub_que
        )

        self.pub_que_dict[target_joint_channel] = Queue(maxsize=1)
        self.publisher_dict[target_joint_channel] = JointMeasPublisher(
            self._lcm_instance,
            target_joint_channel,
            self.pub_que_dict[target_joint_channel],
        )

        self.pub_que_dict[traj_channel] = Queue(maxsize=1)
        self.publisher_dict[traj_channel] = JointTrajPublisher(
            self._lcm_instance,
            traj_channel,
            self.pub_que_dict[traj_channel],
        )

        if status_channel:
            self.pub_que_dict[status_channel] = Queue(maxsize=1)
            self.publisher_dict[status_channel] = P2PStatusPublisher(
                self._lcm_instance,
                status_channel,
                self.pub_que_dict[status_channel],
            )

        if palm_pose_channel:
            self.pub_que_dict[palm_pose_channel] = Queue(maxsize=1)
            self.publisher_dict[palm_pose_channel] = SE3PosePublisher(
                self._lcm_instance,
                palm_pose_channel,
                self.pub_que_dict[palm_pose_channel],
            )
