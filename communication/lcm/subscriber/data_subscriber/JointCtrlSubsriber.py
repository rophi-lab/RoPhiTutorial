import numpy as np
from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.JointCtrlData import JointCtrlData
from lcm_type.joint.joint_ctrl_t import joint_ctrl_t


class JointCtrlSubscriber(BaseDataSubscriber):
    """
    JointCtrlSubscriber is a class that subscribes to joint control data from LCM.
    It inherits from the BaseDataSubscriber class and implements the
    handle_message method to process incoming messages.
    """

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue, num_joints: int):
        super().__init__(lcm_instance)
        self.data_queue = data_queue
        self.num_joints = num_joints

    def _callback(self, channel: str, data):
        """
        Callback function for the subscriber.
        """
        msg = joint_ctrl_t.decode(data)
        joint_ctrl_data = JointCtrlData(num_joints=self.num_joints)
        joint_ctrl_data.set_data(
            msg.timestamp,
            np.array(msg.q_des),
            np.array(msg.qd_des),
            np.array(msg.tau_ff),
            np.array(msg.kp),
            np.array(msg.kd),
        )
        joint_ctrl_data.set_valid(msg.valid)
        self.data_queue.put(joint_ctrl_data)
