from queue import Queue
from typing import List

from communication.lcm.publisher.BasePubManager import BasePubManager
from communication.lcm.publisher.data_publisher.MaskPublisher import MaskPublisher


class MaskPubManager(BasePubManager):
    """Publishes one MaskData stream per mask channel (one per camera view)."""

    def __init__(self, *args, mask_channels: List[str] = ("mask",), **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(mask_channels, str):
            mask_channels = [mask_channels]
        for ch in mask_channels:
            self.pub_que_dict[ch] = Queue()
            self.publisher_dict[ch] = MaskPublisher(
                self._lcm_instance, ch, self.pub_que_dict[ch]
            )
