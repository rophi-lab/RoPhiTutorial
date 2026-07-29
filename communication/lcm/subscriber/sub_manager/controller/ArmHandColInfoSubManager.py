"""Controller subscriber manager: arm/hand joints + robot/static collision info."""

from queue import Queue

from communication.lcm.subscriber.BaseSubManager import BaseControllerSubManager
from communication.lcm.subscriber.data_subscriber.ClockSubscriber import ClockSubscriber
from communication.lcm.subscriber.data_subscriber.ColInfoSubscriber import (
    ColInfoSubscriber,
)
from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)
from communication.lcm.subscriber.data_subscriber.P2PGainsCmdSubscriber import (
    P2PGainsCmdSubscriber,
)


class ArmHandColInfoSubManager(BaseControllerSubManager):
    """Subscribe to arm/hand joint measurements and collision geometry channels.

    Queues
    ------
    ``intr_sub_que_dict``
        Arm and hand ``JointMeasData``, robot (link-local) ``ColInfoData``,
        and optionally ``P2PGainsCmdData`` (Viser gain/friction cmds).
    ``extr_sub_que_dict``
        Static world ``ColInfoData`` (floor / walls).
    ``time_sub_que``
        Simulation clock.
    """

    def __init__(
        self,
        arm_joint_meas_channel: str = "sw_flexiv_arm_joint_meas",
        hand_joint_meas_channel: str = "sw_robotis_5F_hand_joint_meas",
        static_col_info_channel: str = "sw_flexiv_arm_hand_static_col_info",
        robot_col_info_channel: str = "sw_flexiv_arm_hand_robot_col_info",
        gains_cmd_channel: str = "",
        num_arm_joints: int = 7,
        num_hand_joints: int = 20,
        num_joints: int = 27,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_arm_joints = num_arm_joints
        self.num_hand_joints = num_hand_joints
        self.num_joints = num_joints

        # --- queues ---
        self.data_queue_dict["intr_sub_que_dict"][arm_joint_meas_channel] = Queue()
        self.data_queue_dict["intr_sub_que_dict"][hand_joint_meas_channel] = Queue()
        self.data_queue_dict["intr_sub_que_dict"][robot_col_info_channel] = Queue()
        self.data_queue_dict["extr_sub_que_dict"][static_col_info_channel] = Queue()

        # --- joint measurements ---
        JointMeasSubscriber(
            self._lcm_instance,
            self.data_queue_dict["intr_sub_que_dict"][arm_joint_meas_channel],
            num_joints=self.num_arm_joints,
        ).subscribe(arm_joint_meas_channel)

        JointMeasSubscriber(
            self._lcm_instance,
            self.data_queue_dict["intr_sub_que_dict"][hand_joint_meas_channel],
            num_joints=self.num_hand_joints,
        ).subscribe(hand_joint_meas_channel)

        # --- collision geometry ---
        # Robot link primitives (intrinsic; move with the robot).
        ColInfoSubscriber(
            self._lcm_instance,
            self.data_queue_dict["intr_sub_que_dict"][robot_col_info_channel],
        ).subscribe(robot_col_info_channel)

        # Static cell obstacles (extrinsic; world-fixed).
        ColInfoSubscriber(
            self._lcm_instance,
            self.data_queue_dict["extr_sub_que_dict"][static_col_info_channel],
        ).subscribe(static_col_info_channel)

        # Optional Viser → controller gain / friction commands.
        if gains_cmd_channel:
            self.data_queue_dict["intr_sub_que_dict"][gains_cmd_channel] = Queue()
            P2PGainsCmdSubscriber(
                self._lcm_instance,
                self.data_queue_dict["intr_sub_que_dict"][gains_cmd_channel],
            ).subscribe(gains_cmd_channel)

        # --- clock ---
        ClockSubscriber(
            self._lcm_instance, self.data_queue_dict["time_sub_que"]
        ).subscribe("sim_clock")
