from queue import Queue

import lcm
import numpy as np

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.JointTrajData import JointTrajData
from lcm_type.joint.joint_traj_t import joint_traj_t


class JointTrajPublisher(BaseDataPublisher):
    """Publish JointTrajData as joint.joint_traj_t."""

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, JointTrajData):
            return
        timestamp, t, q, qd = data.get_data()
        msg = joint_traj_t()
        msg.timestamp = float(timestamp)
        msg.num_joints = int(q.shape[1])
        msg.num_points = int(q.shape[0])
        msg.t = np.asarray(t, dtype=np.float32).tolist()
        msg.q = np.asarray(q, dtype=np.float32).tolist()
        msg.qd = np.asarray(qd, dtype=np.float32).tolist()
        self._lcm_instance.publish(self._channel, msg.encode())
