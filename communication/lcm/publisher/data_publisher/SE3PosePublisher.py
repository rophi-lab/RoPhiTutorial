from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.SE3PoseData import SE3PoseData
from lcm_type.pose.se3_pose_t import se3_pose_t


class SE3PosePublisher(BaseDataPublisher):
    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, SE3PoseData):
            return
        msg = se3_pose_t()
        msg.timestamp = float(data.timestamp)
        msg.position = data.position.astype(float).tolist()
        msg.quat_wxyz = data.quat_wxyz.astype(float).tolist()
        self._lcm_instance.publish(self._channel, msg.encode())
