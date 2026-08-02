from queue import Queue

from communication.lcm.publisher.BasePubManager import BasePubManager
from communication.lcm.publisher.data_publisher.CameraInfoPublisher import (
    CameraInfoPublisher,
)
from communication.lcm.publisher.data_publisher.RGBDPublisher import RGBDPublisher


class RealSensePubManager(BasePubManager):
    """
    RealSensePubManager class to manage the RealSense camera publisher.
    """

    def __init__(self, realsense: dict, max_pub_freq: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.publisher_dict = {}
        self.pub_que_dict = {}
        for key, channel_name in realsense.items():
            self.pub_que_dict[channel_name] = Queue()
            self.pub_que_dict[channel_name + "_info"] = Queue()

            self.publisher_dict[key] = RGBDPublisher(
                self._lcm_instance,
                channel_name,
                self.pub_que_dict[channel_name],
            )
            self.publisher_dict[channel_name + "_info"] = CameraInfoPublisher(
                self._lcm_instance,
                channel_name + "_info",
                self.pub_que_dict[channel_name + "_info"],
            )
