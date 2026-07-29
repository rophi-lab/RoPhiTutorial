from queue import Queue

from communication.lcm.publisher.data_publisher.JointMeasPublisher import (
    JointMeasPublisher,
)
from communication.lcm.publisher.BasePubManager import BaseEnvPubManager


class FlexivArmHandPubManager(BaseEnvPubManager):
    """Env-side publisher for the Flexiv arm + Robotis 5F hand.

    Publishes two joint_meas_t streams: the 7-DoF arm and the 20-DoF hand, on
    separate channels. Mirrors FlexivArmPubManager (arm-only) with a second
    channel for the hand. The channel names must match the meas channels the
    SimFlexivArm5FHand platform publishes to.
    """

    def __init__(
        self,
        *args,
        arm_joint_meas_channel: str = "hw_flexiv_arm_joint_meas",
        hand_joint_meas_channel: str = "hw_robotis_5F_hand_joint_meas",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pub_que_dict["intr_pub_que_dict"] = {
            arm_joint_meas_channel: Queue(),
            hand_joint_meas_channel: Queue(),
        }

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
