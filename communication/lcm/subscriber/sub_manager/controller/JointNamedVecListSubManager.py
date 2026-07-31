from queue import Queue

from communication.lcm.subscriber.BaseSubManager import BaseControllerSubManager
from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)

from communication.lcm.subscriber.data_subscriber.ClockSubscriber import ClockSubscriber
from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)


class JointNamedVecListSubManager(BaseControllerSubManager):
    """
    This class manages the subscribers for the JointNamedVecListSubManager.
    """

    def __init__(
        self,
        joint_meas_channel: str = "joint_meas",
        named_vec_list_channel: str = "named_vec_list",
        num_joints: int = 7,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.data_queue_dict["intr_sub_que_dict"][joint_meas_channel] = Queue()
        self.data_queue_dict["extr_sub_que_dict"][named_vec_list_channel] = Queue()

        self.num_joints = num_joints

        joint_meas_sub = JointMeasSubscriber(
            self._lcm_instance,
            self.data_queue_dict["intr_sub_que_dict"][joint_meas_channel],
            num_joints=self.num_joints,
        )
        joint_meas_sub.subscribe(joint_meas_channel)

        named_vec_list_sub = NamedVecListSubscriber(
            self._lcm_instance,
            self.data_queue_dict["extr_sub_que_dict"][named_vec_list_channel],
        )
        named_vec_list_sub.subscribe(named_vec_list_channel)

        clock_sub = ClockSubscriber(
            self._lcm_instance, self.data_queue_dict["time_sub_que"]
        )
        clock_sub.subscribe("sim_clock")
