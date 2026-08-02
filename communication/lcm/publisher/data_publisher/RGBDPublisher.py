from queue import Queue
import time

import numpy as np

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.RGBDData import RGBDData
from lcm_type.vision.rgbd_t import rgbd_t
from utils.communication.lcm.image_conversion import (
    CHANNEL_TYPE_TO_DTYPE,
    DTYPE_TO_CHANNEL_TYPE,
    pack_image_to_bytes,
)


class RGBDPublisher(BaseDataPublisher):
    """
    RGBDPublisher is a class that inherits from BaseDataPublisher.
    It is responsible for publishing RGBD data on a specific LCM channel.
    """

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)
        self._exe_t = time.time()
        self._lst_data_t = time.time()

    def check_que_and_publish(self):
        """
        Check if the queue is not empty and publish the data to the LCM channel.
        """
        # print("function call time: ", time.time() - self._exe_t)
        # self._exe_t = time.time()
        if not self._data_que.empty():
            # print("checking time: ", time.time() - self._exe_t)
            # print("rgbd publishing time: ", time.time() - self._lst_data_t)
            self._lst_data_t = time.time()

            data = self._data_que.get()
            if isinstance(data, RGBDData):
                t, rgb_channel_type, rgb_image, depth_channel_type, depth_image = (
                    data.get_data()
                )
                msg = rgbd_t()
                msg.timestamp = t
                msg.height = data.height
                msg.width = data.width

                if rgb_image.ndim == 3:
                    msg.num_rgb_channels = rgb_image.shape[2]
                else:
                    msg.num_rgb_channels = 1

                # RGB
                msg.rgb_channel_type = rgb_channel_type
                rgb_dtype = CHANNEL_TYPE_TO_DTYPE[rgb_channel_type]
                msg.rgb_size = np.size(rgb_image) * np.dtype(rgb_dtype).itemsize
                msg.rgb_image = pack_image_to_bytes(rgb_image, rgb_channel_type)

                # depth
                msg.depth_channel_type = depth_channel_type
                depth_dtype = CHANNEL_TYPE_TO_DTYPE[depth_channel_type]
                msg.depth_size = np.size(depth_image) * np.dtype(depth_dtype).itemsize
                msg.depth_image = pack_image_to_bytes(depth_image, depth_channel_type)

                # Publish the message
                self._lcm_instance.publish(self._channel, msg.encode())
                # print("processing time: ", time.time() - self._exe_t)
