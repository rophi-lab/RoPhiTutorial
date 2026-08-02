from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.MaskData import MaskData
from lcm_type.vision.mask_t import mask_t
from utils.communication.lcm.image_conversion import unpack_image_from_bytes


class MaskSubscriber(BaseDataSubscriber):
    """Subscribes to mask_t and pushes MaskData onto a queue."""

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = mask_t.decode(data)
        mask_image = unpack_image_from_bytes(
            msg.mask_image,
            height=msg.height,
            width=msg.width,
            num_channels=1,
            channel_type=msg.channel_type,
        )
        mask_data = MaskData(
            height=msg.height,
            width=msg.width,
            channel_type=msg.channel_type,
            view_id=msg.view_id,
            name="mask_data",
        )
        mask_data.set_data(
            t=msg.timestamp,
            view_id=msg.view_id,
            mask_image=mask_image,
            channel_type=msg.channel_type,
        )
        self.data_queue.put_nowait(mask_data)
