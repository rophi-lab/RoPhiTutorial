from queue import Queue

from communication.lcm.subscriber.BaseSubManager import BaseControllerSubManager
from communication.lcm.subscriber.data_subscriber.ClockSubscriber import ClockSubscriber
from communication.lcm.subscriber.data_subscriber.ImpGainsCmdSubscriber import (
    ImpGainsCmdSubscriber,
)
from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)


class ArmHandSubManager(BaseControllerSubManager):
    """Arm / hand joint measurements (+ optional impedance gains cmd)."""

    def __init__(
        self,
        arm_joint_meas_channel: str = "sw_flexiv_arm_joint_meas",
        hand_joint_meas_channel: str = "sw_robotis_5F_hand_joint_meas",
        gains_cmd_channel: str = "",
        num_arm_joints: int = 7,
        num_hand_joints: int = 8,
        num_joints: int = 15,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data_queue_dict["intr_sub_que_dict"][arm_joint_meas_channel] = Queue()
        self.data_queue_dict["intr_sub_que_dict"][hand_joint_meas_channel] = Queue()

        self.num_arm_joints = num_arm_joints
        self.num_hand_joints = num_hand_joints
        self.num_joints = num_joints

        arm_joint_meas_sub = JointMeasSubscriber(
            self._lcm_instance,
            self.data_queue_dict["intr_sub_que_dict"][arm_joint_meas_channel],
            num_joints=self.num_arm_joints,
        )
        arm_joint_meas_sub.subscribe(arm_joint_meas_channel)

        hand_joint_meas_sub = JointMeasSubscriber(
            self._lcm_instance,
            self.data_queue_dict["intr_sub_que_dict"][hand_joint_meas_channel],
            num_joints=self.num_hand_joints,
        )
        hand_joint_meas_sub.subscribe(hand_joint_meas_channel)

        if gains_cmd_channel:
            self.data_queue_dict["intr_sub_que_dict"][gains_cmd_channel] = Queue()
            ImpGainsCmdSubscriber(
                self._lcm_instance,
                self.data_queue_dict["intr_sub_que_dict"][gains_cmd_channel],
            ).subscribe(gains_cmd_channel)

        clock_sub = ClockSubscriber(
            self._lcm_instance, self.data_queue_dict["time_sub_que"]
        )
        clock_sub.subscribe("sim_clock")
