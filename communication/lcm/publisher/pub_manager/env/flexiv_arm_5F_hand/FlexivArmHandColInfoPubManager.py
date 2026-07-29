from queue import Queue

from communication.lcm.publisher.data_publisher.JointMeasPublisher import (
    JointMeasPublisher,
)
from communication.lcm.publisher.data_publisher.ColInfoPublisher import (
    ColInfoPublisher,
)
from communication.lcm.publisher.BasePubManager import BaseEnvPubManager


class FlexivArmHandColInfoPubManager(BaseEnvPubManager):
    """Env-side publisher for the Flexiv arm + Robotis 5F hand WITH collision
    info. Extends FlexivArmHandPubManager (arm + hand joint_meas) with two
    ColInfoData streams for collision-aware control:

    - ``robot_col_info_channel`` (intr): per-link collision primitives that
      follow the robot (the controller does FK to place them).
    - ``static_col_info_channel`` (extr): the static cell walls.

    Routing (intr vs extr) matches BrlLabWithFixtures / the collision-aware
    controller's sub_manager, which reads robot col info from the intr queue and
    static col info from the extr queue.
    """

    def __init__(
        self,
        *args,
        arm_joint_meas_channel: str = "hw_flexiv_arm_joint_meas",
        hand_joint_meas_channel: str = "hw_robotis_5F_hand_joint_meas",
        static_col_info_channel: str = "hw_flexiv_arm_hand_static_col_info",
        robot_col_info_channel: str = "hw_flexiv_arm_hand_robot_col_info",
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
