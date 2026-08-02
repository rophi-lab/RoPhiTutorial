from queue import Queue
from typing import List

from communication.lcm.subscriber.BaseSubManager import BasePerceptionSubManager
from communication.lcm.subscriber.data_subscriber.RGBDSubscriber import RGBDSubscriber
from communication.lcm.subscriber.data_subscriber.CameraInfoSubscriber import (
    CameraInfoSubscriber,
)


class MultiRGBDSubManager(BasePerceptionSubManager):
    """
    Manages RGBD data subscription for multiple camera views.

    Generalizes SingleRGBDSubManager: for each channel in ``rgbd_channels`` it
    creates a data queue ``{channel}`` and an info queue ``{channel}_info`` and
    subscribes an RGBDSubscriber / CameraInfoSubscriber to each. All queues live
    in a single ``extr_sub_que_dict`` so a multi-view perception module can pull
    each view independently.
    """

    def __init__(self, rgbd_channels: List[str], *args, **kwargs):
        """
        @param[in] rgbd_channels: list of RGBD channel names, one per camera view.
        """
        super().__init__(*args, **kwargs)

        if isinstance(rgbd_channels, str):
            # Tolerate a single channel passed as a bare string.
            rgbd_channels = [rgbd_channels]
        self._rgbd_channels = list(rgbd_channels)

        extr = {}
        self._rgbd_subs = []
        self._camera_info_subs = []
        for rgbd_channel in self._rgbd_channels:
            extr[rgbd_channel] = Queue()
            extr[rgbd_channel + "_info"] = Queue()

            rgbd_sub = RGBDSubscriber(
                lcm_instance=self._lcm_instance,
                data_queue=extr[rgbd_channel],
            )
            rgbd_sub.subscribe(rgbd_channel)
            self._rgbd_subs.append(rgbd_sub)

            camera_info_sub = CameraInfoSubscriber(
                lcm_instance=self._lcm_instance,
                data_queue=extr[rgbd_channel + "_info"],
            )
            camera_info_sub.subscribe(rgbd_channel + "_info")
            self._camera_info_subs.append(camera_info_sub)

        self.data_queue_dict["extr_sub_que_dict"] = extr
