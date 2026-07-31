"""Env subscriber: joint control + one NamedVecList (e.g. contact forces)."""

from queue import Queue

from communication.lcm.subscriber.BaseSubManager import BaseEnvSubManeager
from communication.lcm.subscriber.data_subscriber.JointCtrlSubsriber import (
    JointCtrlSubscriber,
)
from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)


class JointNamedVecListSubManager(BaseEnvSubManeager):
    """Subscribe to JointCtrl plus an extrinsic NamedVecList channel."""

    def __init__(
        self,
        ctrl_channel: str = "sim_joint_ctrl",
        named_vec_list_channel: str = "sim_contact_forces_in_world",
        num_joints: int = 20,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_joints = num_joints

        ctrl_sub = JointCtrlSubscriber(
            self._lcm_instance,
            self.data_queue_dict["ctrl_sub_que"],
            num_joints=self.num_joints,
        )
        ctrl_sub.subscribe(ctrl_channel)

        self.data_queue_dict["extr_sub_que_dict"][named_vec_list_channel] = Queue()
        named_sub = NamedVecListSubscriber(
            self._lcm_instance,
            self.data_queue_dict["extr_sub_que_dict"][named_vec_list_channel],
        )
        named_sub.subscribe(named_vec_list_channel)
