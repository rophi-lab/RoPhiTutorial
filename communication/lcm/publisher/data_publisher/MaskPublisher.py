from queue import Queue

import numpy as np

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.MaskData import MaskData
from lcm_type.vision.mask_t import mask_t
from utils.communication.lcm.image_conversion import (
    CHANNEL_TYPE_TO_DTYPE,
    pack_image_to_bytes,
)


class MaskPublisher(BaseDataPublisher):
    """Publishes MaskData (a single-object mask stamped with the source RGB
    frame timestamp and a view id) on an LCM channel."""

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, MaskData):
            return
        t, view_id, mask_image = data.get_data()

        msg = mask_t()
        msg.timestamp = t
        msg.view_id = int(view_id)
        msg.height = data.height
        msg.width = data.width
        msg.channel_type = data.channel_type
        dtype = CHANNEL_TYPE_TO_DTYPE[data.channel_type]
        msg.mask_size = int(np.size(mask_image) * np.dtype(dtype).itemsize)
        msg.mask_image = pack_image_to_bytes(mask_image, data.channel_type)

        self._lcm_instance.publish(self._channel, msg.encode())
