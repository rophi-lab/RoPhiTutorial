from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.Point3DData import Point3DData
from lcm_type.geometry.point_set_t import point_set_t


class Point3DPublisher(BaseDataPublisher):
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
            if isinstance(data, Point3DData):
                # convert the data to the LCM message type
                msg = point_set_t()
                msg.timestamp = data.get_time()
                msg.num_points = data.get_num_points()
                msg.position = data.get_position().tolist()
                # publish the message to the LCM channel
                self._lcm_instance.publish(self._channel, msg.encode())
            else:
                print("Data is not valid, not publishing.")
