from queue import Queue
from typing import List

from communication.lcm.subscriber.BaseSubManager import BasePerceptionSubManager
from communication.lcm.subscriber.data_subscriber.RGBDSubscriber import RGBDSubscriber
from communication.lcm.subscriber.data_subscriber.CameraInfoSubscriber import (
    CameraInfoSubscriber,
)
from communication.lcm.subscriber.data_subscriber.MaskSubscriber import MaskSubscriber


class MultiRGBDMaskSubManager(BasePerceptionSubManager):
    """RGBD (+ camera info) per view AND a SAM2 mask stream per view.

    Used by the split-pipeline FoundationPose node: it subscribes each RGBD
    channel (``{ch}`` + ``{ch}_info``) and its matching mask channel so masks and
    frames can be time-matched for registration. All queues live in
    ``extr_sub_que_dict``.
    """

    def __init__(
        self,
        rgbd_channels: List[str],
        mask_channels: List[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(rgbd_channels, str):
            rgbd_channels = [rgbd_channels]
        self._rgbd_channels = list(rgbd_channels)
        if mask_channels is None:
            mask_channels = [ch + "_mask" for ch in self._rgbd_channels]
        if isinstance(mask_channels, str):
            mask_channels = [mask_channels]
        if len(mask_channels) != len(self._rgbd_channels):
            raise ValueError(
                f"mask_channels ({len(mask_channels)}) must match rgbd_channels "
                f"({len(self._rgbd_channels)})"
            )
        self._mask_channels = list(mask_channels)

        extr = {}
        self._rgbd_subs = []
        self._camera_info_subs = []
        self._mask_subs = []
        for rgbd_channel, mask_channel in zip(
            self._rgbd_channels, self._mask_channels
        ):
            extr[rgbd_channel] = Queue()
            extr[rgbd_channel + "_info"] = Queue()
            extr[mask_channel] = Queue()

            rgbd_sub = RGBDSubscriber(
                lcm_instance=self._lcm_instance, data_queue=extr[rgbd_channel]
            )
            rgbd_sub.subscribe(rgbd_channel)
            self._rgbd_subs.append(rgbd_sub)

            info_sub = CameraInfoSubscriber(
                lcm_instance=self._lcm_instance,
                data_queue=extr[rgbd_channel + "_info"],
            )
            info_sub.subscribe(rgbd_channel + "_info")
            self._camera_info_subs.append(info_sub)

            mask_sub = MaskSubscriber(
                lcm_instance=self._lcm_instance, data_queue=extr[mask_channel]
            )
            mask_sub.subscribe(mask_channel)
            self._mask_subs.append(mask_sub)

        self.data_queue_dict["extr_sub_que_dict"] = extr
