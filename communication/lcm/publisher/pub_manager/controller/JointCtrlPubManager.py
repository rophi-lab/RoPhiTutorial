from queue import Queue

import lcm

from communication.lcm.publisher.BasePubManager import BaseCtrlPubManager
from communication.lcm.publisher.data_publisher.JointCtrlPublisher import (
    JointCtrlPublisher,
)


class JointCtrlPubManager(BaseCtrlPubManager):
    """
    Class for managing the publisher for the BRL Arm trajectory tracking.
    """

    def __init__(self, ctrl_channel: str = "brl_arm_joint_ctrl", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.publisher_dict[ctrl_channel] = JointCtrlPublisher(
            self._lcm_instance, ctrl_channel, self.ctrl_pub_que
        )
