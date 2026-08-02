import numpy as np
from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.CameraInfoData import CameraInfoData
from lcm_type.vision.camera_info_t import camera_info_t


class CameraInfoSubscriber(BaseDataSubscriber):
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

        self.intrinsic = np.eye(3)
        self.extrinsic = np.eye(4)
        self.depth_factor = 1

    def _callback(self, channel: str, data):
        """
        Callback function for the subscriber.
        """
        msg = camera_info_t.decode(data)
        t = msg.timestamp
        height = msg.height
        width = msg.width

        self.intrinsic[0, 0] = msg.fx
        self.intrinsic[1, 1] = msg.fy
        self.intrinsic[0, 2] = msg.cx
        self.intrinsic[1, 2] = msg.cy

        self.extrinsic[:3, :] = np.array(msg.extrinsic).reshape(3, 4)

        self.depth_factor = msg.depth_factor

        self.fixed = msg.fixed
        self.attached_body = msg.attached_body

        camera_info_data = CameraInfoData(
            height=height,
            width=width,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic,
            fixed=self.fixed,
            attached_body=self.attached_body,
            depth_factor=self.depth_factor
        )

        self.data_queue.put(camera_info_data)
