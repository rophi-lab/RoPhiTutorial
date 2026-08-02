"""Environment publisher manager: RGBD camera (+ info) and optional GT pose."""

from queue import Queue

from communication.lcm.publisher.BasePubManager import BaseEnvPubManager
from communication.lcm.publisher.data_publisher.CameraInfoPublisher import (
    CameraInfoPublisher,
)
from communication.lcm.publisher.data_publisher.NamedVecListPublisher import (
    NamedVecListPublisher,
)
from communication.lcm.publisher.data_publisher.RGBDPublisher import RGBDPublisher


class CamOnlyPubManager(BaseEnvPubManager):
    """Publish RGBD / camera_info and an optional NamedVecList GT object pose."""

    def __init__(
        self,
        *args,
        upper_camera_channel: str = "upper_camera",
        named_vec_list_channel: str = "sim_obj_pose_bb2world",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pub_que_dict["intr_pub_que_dict"] = {}
        self.pub_que_dict["extr_pub_que_dict"] = {
            upper_camera_channel: Queue(),
            upper_camera_channel + "_info": Queue(),
            named_vec_list_channel: Queue(),
        }

        for key, item in self.pub_que_dict["extr_pub_que_dict"].items():
            if key.endswith("_info"):
                self.publisher_dict[key] = CameraInfoPublisher(
                    self._lcm_instance, key, item
                )
            elif key == named_vec_list_channel:
                self.publisher_dict[key] = NamedVecListPublisher(
                    self._lcm_instance, key, item
                )
            else:
                self.publisher_dict[key] = RGBDPublisher(
                    self._lcm_instance, key, item
                )
