import numpy as np
from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.JointMeasData import JointMeasData
from lcm_type.joint.joint_meas_t import joint_meas_t


class JointMeasSubscriber(BaseDataSubscriber):
    """
    JointMeasSubscriber is a class that subscribes to joint meas data from LCM.
    It inherits from the BaseDataSubscriber class and implements the
    handle_message method to process incoming messages.
    """

    def __init__(
        self,
        lcm_instance: lcm.LCM,
        data_queue: Queue,
        num_joints: int,
    ):
        super().__init__(lcm_instance)
        self.data_queue = data_queue
        self.num_joints = num_joints

    def _callback(self, channel: str, data):
        """
        Callback function for the subscriber.
        """
        msg = joint_meas_t.decode(data)
        joint_meas_data = JointMeasData(num_joints=self.num_joints)
        joint_meas_data.set_data(
            msg.timestamp,
            np.array(msg.q),
            np.array(msg.qd),
            np.array(msg.tau),
        )
        self.data_queue.put(joint_meas_data)
