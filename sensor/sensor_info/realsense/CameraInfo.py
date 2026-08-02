import numpy as np
import copy

from scipy.spatial.transform import Rotation as R

from sensor.BaseSensorInfo import BaseSensorInfo
from data_type.basic_types.CameraInfoData import CameraInfoData


class CameraInfo(BaseSensorInfo):
    """
    Camera information class.
    """

    def __init__(
        self,
        name: str,
        serial: int,
        fixed: bool,
        attached_body: str,
        height: int,
        width: int,
        frequency: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_factor: float,
        tf_link_to_cam: dict,
        channel_name: str,
        *args,
        **kwargs
    ):
        super().__init__(name, *args, **kwargs)

        self.serial = serial
        self.fixed = fixed
        self.attached_body = attached_body

        self.T_link_to_cam = np.eye(4)
        self.T_link_to_cam[0:3, 0:3] = R.from_quat(
            tf_link_to_cam["quaternion"]
        ).as_matrix()
        self.T_link_to_cam[0:3, 3] = np.array(tf_link_to_cam["translation"])

        self.image_width = width
        self.image_height = height
        self.frequency = frequency

        self.data = CameraInfoData(
            height=height,
            width=width,
            intrinsic=np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ]
            ),
            extrinsic=self.T_link_to_cam,
            fixed=fixed,
            attached_body=attached_body,
            depth_factor=depth_factor,
            name=channel_name,
        )
        print(self.data.get_depth_factor())

    def update_extrinsic(self, fk_data):
        """
        Update the extrinsic matrix.
        """
        if self.fixed:
            return

        ## TODO: COME BACK WHEN WE ATTACH A CAMERA TO THE WRIST

    def update(self):
        self.pub_data_queue.put(copy.deepcopy(self.data))
