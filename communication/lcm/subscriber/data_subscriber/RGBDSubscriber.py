import numpy as np
from queue import Queue
import time

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.RGBDData import RGBDData
from lcm_type.vision.rgbd_t import rgbd_t

from utils.communication.lcm.image_conversion import unpack_image_from_bytes


class RGBDSubscriber(BaseDataSubscriber):
    """
    RGBDSubscriber is a class that subscribes to RGBD data from LCM.
    """

    def __init__(
        self,
        lcm_instance: lcm.LCM,
        data_queue: Queue,
    ):
        super().__init__(lcm_instance)
        self.data_queue = data_queue
        # self._last_t = time.time()

    def _callback(self, channel: str, data):
        """
        Callback function for the subscriber.
        """
        # print("callback time: ", time.time() - self._last_t)
        # self._last_t = time.time()
        msg = rgbd_t.decode(data)
        t = msg.timestamp
        height = msg.height
        width = msg.width

        rgb_channel_type = msg.rgb_channel_type
        num_rgb_channels = msg.num_rgb_channels
        depth_channel_type = msg.depth_channel_type

        rgbd_data = RGBDData(
            height=height,
            width=width,
            rgb_channel_type=rgb_channel_type,  # uint8
            depth_channel_type=depth_channel_type,  # uint16
            name="rgbd_data",
        )

        rgb_image = unpack_image_from_bytes(
            msg.rgb_image,
            height=height,
            width=width,
            num_channels=num_rgb_channels,
            channel_type=rgb_channel_type,
        )
        depth_image = unpack_image_from_bytes(
            msg.depth_image,
            height=height,
            width=width,
            num_channels=1,
            channel_type=depth_channel_type,
        )

        rgbd_data.set_data(
            t, rgb_channel_type, rgb_image, depth_channel_type, depth_image
        )
        self.data_queue.put_nowait(rgbd_data)
        # print(self.data_queue.qsize())
        # print(time.time() - self._last_t)
