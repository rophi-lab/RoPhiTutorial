from queue import Queue

import lcm
import numpy as np

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.Pose3DData import Pose3DData
from lcm_type.pose.pose3d_t import pose3d_t


class Pose3DSubscriber(BaseDataSubscriber):
    """Subscribe to pose.pose3d_t and push Pose3DData onto a queue."""

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = pose3d_t.decode(data)
        pose = Pose3DData(name=channel)
        pose.set_data(
            float(msg.timestamp),
            np.asarray(msg.position, dtype=np.float64),
            np.asarray(msg.velocity, dtype=np.float64),
            np.asarray(msg.tip, dtype=np.float64),
        )
        self.data_queue.put(pose)
