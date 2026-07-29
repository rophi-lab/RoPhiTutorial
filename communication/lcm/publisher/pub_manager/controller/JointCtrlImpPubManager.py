from queue import Queue

from communication.lcm.publisher.BasePubManager import BaseCtrlPubManager
from communication.lcm.publisher.data_publisher.ImpStatusPublisher import (
    ImpStatusPublisher,
)
from communication.lcm.publisher.data_publisher.JointCtrlPublisher import (
    JointCtrlPublisher,
)


class JointCtrlImpPubManager(BaseCtrlPubManager):
    """Publish joint torque commands + impedance status for Viser."""

    def __init__(
        self,
        ctrl_channel: str = "sw_flexiv_arm_hand_joint_ctrl",
        status_channel: str = "sw_imp_status",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.publisher_dict[ctrl_channel] = JointCtrlPublisher(
            self._lcm_instance, ctrl_channel, self.ctrl_pub_que
        )

        if status_channel:
            self.pub_que_dict[status_channel] = Queue(maxsize=1)
            self.publisher_dict[status_channel] = ImpStatusPublisher(
                self._lcm_instance,
                status_channel,
                self.pub_que_dict[status_channel],
            )
