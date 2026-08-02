from queue import Queue

from communication.lcm.subscriber.BaseSubManager import BasePerceptionSubManager
from communication.lcm.subscriber.data_subscriber.RGBDSubscriber import RGBDSubscriber
from communication.lcm.subscriber.data_subscriber.CameraInfoSubscriber import (
    CameraInfoSubscriber,
)


class SingleRGBDSubManager(BasePerceptionSubManager):
    """
    This class manages the RGBD data subscription.
    It inherits from the BaseSubManager class.
    """

    def __init__(self, rgbd_channel: str = "RGBD", *args, **kwargs):
        """
        Initialize the SingleRGBDSubManager.
        @param[in] lcm_instance: The LCM instance to use for communication.
        @param[in] rgbd_channel: The channel name for the RGBD data.
        """
        super().__init__(*args, **kwargs)
        self._rgbd_channel = rgbd_channel

        self.data_queue_dict["extr_sub_que_dict"] = {
            rgbd_channel: Queue(),
            rgbd_channel + "_info": Queue(),
        }

        self._rgbd_sub = RGBDSubscriber(
            lcm_instance=self._lcm_instance,
            data_queue=self.data_queue_dict["extr_sub_que_dict"][rgbd_channel],
        )
        self._rgbd_sub.subscribe(self._rgbd_channel)

        self._camera_info_sub = CameraInfoSubscriber(
            lcm_instance=self._lcm_instance,
            data_queue=self.data_queue_dict["extr_sub_que_dict"][
                rgbd_channel + "_info"
            ],
        )
        self._camera_info_sub.subscribe(self._rgbd_channel + "_info")
