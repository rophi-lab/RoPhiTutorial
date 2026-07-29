from queue import Queue


from communication.lcm.publisher.data_publisher.JointMeasPublisher import (
    JointMeasPublisher,
)
from communication.lcm.publisher.data_publisher.NamedVecListPublisher import (
    NamedVecListPublisher,
)

from communication.lcm.publisher.BasePubManager import BaseEnvPubManager


class JointNamedVecListPubManager(BaseEnvPubManager):
    """
    Class for managing the publisher for the PushT environment.
    """

    def __init__(
        self,
        *args,
        joint_meas_channel: str = "joint_meas",
        named_vec_list_channel: str = "named_vec_list",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.pub_que_dict["intr_pub_que_dict"] = {joint_meas_channel: Queue()}

        self.pub_que_dict["extr_pub_que_dict"] = {named_vec_list_channel: Queue()}

        self.publisher_dict[joint_meas_channel] = JointMeasPublisher(
            self._lcm_instance,
            joint_meas_channel,
            self.pub_que_dict["intr_pub_que_dict"][joint_meas_channel],
        )

        self.publisher_dict[named_vec_list_channel] = NamedVecListPublisher(
            self._lcm_instance,
            named_vec_list_channel,
            self.pub_que_dict["extr_pub_que_dict"][named_vec_list_channel],
        )
