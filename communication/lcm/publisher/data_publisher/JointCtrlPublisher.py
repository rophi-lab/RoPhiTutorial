from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.JointCtrlData import JointCtrlData
from lcm_type.joint.joint_ctrl_t import joint_ctrl_t


class JointCtrlPublisher(BaseDataPublisher):
    """
    JointMeasPublisher is a class that inherits from BaseDataPublisher.
    It is responsible for publishing joint measurement data on a specific LCM channel.
    """

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        """
        Check if the queue is not empty and publish the data to the LCM channel.
        """
        if not self._data_que.empty():
            data = self._data_que.get()
            if isinstance(data, JointCtrlData):
                t, q_des, qd_des, tau_ff, kp, kd = data.get_data()
                msg = joint_ctrl_t()
                msg.timestamp = t
                msg.q_des = q_des.tolist()
                msg.qd_des = qd_des.tolist()
                msg.tau_ff = tau_ff.tolist()
                msg.kp = kp.tolist()
                msg.kd = kd.tolist()
                msg.num_joints = len(q_des)
                msg.valid = data.get_valid()
                self._lcm_instance.publish(self._channel, msg.encode())
