from queue import Queue

import lcm
import numpy as np

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.Pose3DData import Pose3DData
from lcm_type.pose.pose3d_t import pose3d_t


class Pose3DPublisher(BaseDataPublisher):
    """Publish Pose3DData as pose.pose3d_t on an LCM channel."""

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, Pose3DData):
            print(f"[Pose3DPublisher] Expected Pose3DData, got {type(data)}")
            return
        t, position, velocity, tip = data.get_data()
        msg = pose3d_t()
        msg.timestamp = float(t)
        msg.position = np.asarray(position, dtype=np.float64).reshape(3).tolist()
        msg.velocity = np.asarray(velocity, dtype=np.float64).reshape(3).tolist()
        msg.tip = np.asarray(tip, dtype=np.float64).reshape(3).tolist()
        self._lcm_instance.publish(self._channel, msg.encode())
