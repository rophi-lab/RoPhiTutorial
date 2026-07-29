from queue import Queue

from communication.lcm.publisher.data_publisher.JointMeasPublisher import (
    JointMeasPublisher,
)
from communication.lcm.publisher.BasePubManager import BaseEnvPubManager


class FlexivHandHwPubManager(BaseEnvPubManager):
    """Hardware-side publisher for the Robotis 5F hand only.

    In the combined Flexiv arm + hand hardware platform the ARM measurement is
    published by the C++ bridge on its own LCM instance, so the Python side only
    needs to publish the hand joint_meas. The channel name matches what the sim
    uses (hw_robotis_5F_hand_joint_meas) so controllers/visualizers don't care
    whether they're talking to sim or hardware.
    """

    def __init__(
        self,
        *args,
        hand_joint_meas_channel: str = "hw_robotis_5F_hand_joint_meas",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pub_que_dict["intr_pub_que_dict"] = {hand_joint_meas_channel: Queue()}
        self.publisher_dict[hand_joint_meas_channel] = JointMeasPublisher(
            self._lcm_instance,
            hand_joint_meas_channel,
            self.pub_que_dict["intr_pub_que_dict"][hand_joint_meas_channel],
        )
