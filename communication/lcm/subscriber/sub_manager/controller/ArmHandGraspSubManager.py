"""Controller sub manager: arm/hand + ColInfo + grasp-object SE(3) + physics."""

from queue import Queue

from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from communication.lcm.subscriber.data_subscriber.SE3PoseSubscriber import (
    SE3PoseSubscriber,
)
from communication.lcm.subscriber.sub_manager.controller.ArmHandColInfoSubManager import (
    ArmHandColInfoSubManager,
)


class ArmHandGraspSubManager(ArmHandColInfoSubManager):
    """``arm_hand_joint_col_info`` plus object SE(3) pose and physics."""

    def __init__(
        self,
        object_pose_channel: str = "sw_grasp_object_pose",
        object_physics_channel: str = "sw_grasp_object_physics",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if object_pose_channel:
            self.data_queue_dict["extr_sub_que_dict"][object_pose_channel] = Queue()
            SE3PoseSubscriber(
                self._lcm_instance,
                self.data_queue_dict["extr_sub_que_dict"][object_pose_channel],
            ).subscribe(object_pose_channel)
        if object_physics_channel:
            self.data_queue_dict["extr_sub_que_dict"][object_physics_channel] = Queue()
            NamedVecListSubscriber(
                self._lcm_instance,
                self.data_queue_dict["extr_sub_que_dict"][object_physics_channel],
            ).subscribe(object_physics_channel)
