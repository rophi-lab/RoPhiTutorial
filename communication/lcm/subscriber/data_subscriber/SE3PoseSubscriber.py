from queue import Queue

import numpy as np
import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.SE3PoseData import SE3PoseData
from lcm_type.pose.se3_pose_t import se3_pose_t


class SE3PoseSubscriber(BaseDataSubscriber):
    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = se3_pose_t.decode(data)
        pose = SE3PoseData()
        pose.set_data(
            float(msg.timestamp),
            np.asarray(msg.position, dtype=np.float64),
            np.asarray(msg.quat_wxyz, dtype=np.float64),
        )
        self.data_queue.put(pose)
