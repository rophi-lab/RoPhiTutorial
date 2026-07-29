from queue import Queue

import lcm
import numpy as np

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.JointTrajData import JointTrajData
from lcm_type.joint.joint_traj_t import joint_traj_t


class JointTrajSubscriber(BaseDataSubscriber):
    """Subscribe to joint.joint_traj_t and push JointTrajData onto a queue."""

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = joint_traj_t.decode(data)
        traj = JointTrajData(name=channel)
        traj.set_data(
            float(msg.timestamp),
            np.asarray(msg.t, dtype=np.float64),
            np.asarray(msg.q, dtype=np.float64),
            np.asarray(msg.qd, dtype=np.float64),
        )
        self.data_queue.put(traj)
