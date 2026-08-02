import numpy as np

from data_type.BaseData import BaseData
from utils.communication.lcm.image_conversion import CHANNEL_TYPE_TO_DTYPE


class MaskData(BaseData):
    """A single-object segmentation mask, stamped with the source RGB frame's
    timestamp and the view/camera index it belongs to. Used to ship SAM2 masks
    from a segmentation process to a pose-estimation process; the timestamp lets
    the consumer time-match the mask against a buffered RGBD frame."""

    def __init__(
        self,
        height: int = 0,
        width: int = 0,
        channel_type: int = 1,  # uint8
        view_id: int = 0,
        name: str = "mask_data",
    ):
        super().__init__(name)
        self.height = height
        self.width = width
        self.channel_type = channel_type
        self.view_id = view_id
        self.mask_image = np.zeros(
            (height, width), dtype=CHANNEL_TYPE_TO_DTYPE[self.channel_type]
        )

    def set_data(
        self,
        t: float,
        view_id: int,
        mask_image: np.ndarray,
        channel_type: int = 1,
    ):
        self.timestamp = t
        self.view_id = view_id
        self.channel_type = channel_type
        self.mask_image = mask_image.copy()
        self.height, self.width = mask_image.shape[:2]

    def get_data(self):
        return (
            self.timestamp,
            self.view_id,
            self.mask_image.copy(),
        )

    def get_mask_image(self):
        return self.mask_image.copy()

    def get_view_id(self):
        return self.view_id

    def get_img_dim(self):
        return self.height, self.width
