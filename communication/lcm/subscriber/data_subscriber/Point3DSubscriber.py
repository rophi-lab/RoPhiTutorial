import numpy as np
from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.Point3DData import Point3DData
from lcm_type.geometry.point_set_t import point_set_t


class Point3DSubscriber(BaseDataSubscriber):
    """
    Point3DSubscriber is a class that subscribes to point 3D data from LCM.
    It inherits from the BaseDataSubscriber class and implements the
    handle_message method to process incoming messages.
    """

    def __init__(
        self,
        lcm_instance: lcm.LCM,
        data_queue: Queue,
    ):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        """
        Callback function for the subscriber.
        """
        msg = point_set_t.decode(data)
        num_points = msg.num_points
        point_3d_data = Point3DData(num_points=num_points)
        point_3d_data.set_data(
            msg.timestamp,
            num_points,
            np.array(msg.position).reshape(num_points, 3),
        )
        self.data_queue.put(point_3d_data)
