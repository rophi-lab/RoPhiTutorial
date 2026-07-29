"""Controller sub manager: arm/hand + ColInfo + grasp-object SE(3)."""

from queue import Queue

from communication.lcm.subscriber.data_subscriber.ClockSubscriber import ClockSubscriber
from communication.lcm.subscriber.data_subscriber.ColInfoSubscriber import (
    ColInfoSubscriber,
)
from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)
from communication.lcm.subscriber.data_subscriber.SE3PoseSubscriber import (
    SE3PoseSubscriber,
)
from communication.lcm.subscriber.sub_manager.controller.ArmHandColInfoSubManager import (
    ArmHandColInfoSubManager,
)


class ArmHandGraspSubManager(ArmHandColInfoSubManager):
    """``arm_hand_joint_col_info`` plus extrinsic object SE(3) pose."""

    def __init__(
        self,
        object_pose_channel: str = "sw_grasp_object_pose",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not object_pose_channel:
            return
        self.data_queue_dict["extr_sub_que_dict"][object_pose_channel] = Queue()
        SE3PoseSubscriber(
            self._lcm_instance,
            self.data_queue_dict["extr_sub_que_dict"][object_pose_channel],
        ).subscribe(object_pose_channel)
