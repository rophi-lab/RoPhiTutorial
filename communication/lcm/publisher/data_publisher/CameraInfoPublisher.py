from queue import Queue
import time

import numpy as np

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.CameraInfoData import CameraInfoData
from lcm_type.vision.camera_info_t import camera_info_t


class CameraInfoPublisher(BaseDataPublisher):
    """
    CameraInfoPublisher class to publish camera intrinsic and extrinsic parameters.
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
            if isinstance(data, CameraInfoData):

                (
                    t,
                    height,
                    width,
                    intrinsic,
                    extrinsic,
                    fixed,
                    attached_body,
                    depth_factor,
                ) = data.get_data()
                msg = camera_info_t()
                msg.timestamp = t
                msg.height = height
                msg.width = width
                msg.fx = intrinsic[0, 0]
                msg.fy = intrinsic[1, 1]
                msg.cx = intrinsic[0, 2]
                msg.cy = intrinsic[1, 2]
                msg.extrinsic = extrinsic[:3, :].flatten().tolist()

                msg.fixed = fixed
                msg.attached_body = attached_body

                msg.depth_factor = depth_factor
                # Publish the message
                self._lcm_instance.publish(self._channel, msg.encode())
