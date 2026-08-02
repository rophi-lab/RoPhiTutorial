import numpy as np

from data_type.BaseData import BaseData
from utils.communication.lcm.image_conversion import (
    CHANNEL_TYPE_TO_DTYPE,
    DTYPE_TO_CHANNEL_TYPE,
)


class RGBDData(BaseData):
    def __init__(
        self,
        height: int,
        width: int,
        rgb_channel_type: int = 1,  # uint8
        depth_channel_type: int = 3,  # uint16
        name: str = "rgbd_data",
    ):
        """
        Initialize the RGBDData object.
        """
        super().__init__(name)
        self.width = width
        self.height = height

        self.rgb_channel_type = rgb_channel_type
        self.depth_channel_type = depth_channel_type

        self.rgb_image = np.zeros(
            (height, width, 3), dtype=CHANNEL_TYPE_TO_DTYPE[self.rgb_channel_type]
        )
        self.depth_image = np.zeros(
            (height, width), dtype=CHANNEL_TYPE_TO_DTYPE[self.depth_channel_type]
        )

    def set_data(
        self,
        t: float,
        rgb_channel_type: int,
        rgb_image: np.ndarray,
        depth_channel_type: int,
        depth_image: np.ndarray,
    ):
        """
        Set the RGBD data.
        """
        self.rgb_channel_type = rgb_channel_type
        self.rgb_image = rgb_image.copy()
        self.depth_channel_type = depth_channel_type
        self.depth_image = depth_image.copy()
        self.timestamp = t

    def set_rgb_channel_type(self, rgb_channel_type: int):
        """
        Set the RGB channel type.
        """
        self.rgb_channel_type = rgb_channel_type

    def set_rgb_image(self, rgb_image: np.ndarray):
        """
        Set the RGB image.
        """
        self.rgb_image = rgb_image.copy()

    def set_depth_channel_type(self, depth_channel_type: int):
        """
        Set the depth channel type.
        """
        self.depth_channel_type = depth_channel_type

    def set_depth_image(self, depth_image: np.ndarray):
        """
        Set the depth image.
        """
        self.depth_image = depth_image.copy()

    def set_img_dim(
        self,
        height: int,
        width: int,
    ):
        """
        Set the image dimensions.
        """
        self.width = width
        self.height = height

    def get_data(self):
        """
        Get the RGBD data.
        """
        return (
            self.timestamp,
            self.rgb_channel_type,
            self.rgb_image.copy(),
            self.depth_channel_type,
            self.depth_image.copy(),
        )

    def get_rgb_channel_type(self):
        """
        Get the RGB channel type.
        """
        return self.rgb_channel_type

    def get_rgb_image(self):
        """
        Get the RGB image.
        """
        return self.rgb_image.copy()

    def get_depth_channel_type(self):
        """
        Get the depth channel type.
        """
        return self.depth_channel_type

    def get_depth_image(self):
        """
        Get the depth image.
        """
        return self.depth_image.copy()

    def get_img_dim(self):
        """
        Get the image dimensions.
        """
        return self.height, self.width
