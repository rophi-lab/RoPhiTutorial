import numpy as np
from data_type.BaseData import BaseData


class CameraInfoData(BaseData):
    """
    CameraInfo class to store camera intrinsic and extrinsic parameters."""

    def __init__(
        self,
        height: int,
        width: int,
        intrinsic: np.ndarray = np.eye(3),
        extrinsic: np.ndarray = np.eye(4),
        fixed: bool = False,
        attached_body: str = None,
        depth_factor: float = 1.0,
        name: str = "camera_info",
    ):
        """
        Initialize the CameraInfo object.
        """
        super().__init__(name)
        self.height = height
        self.width = width

        self.fixed = fixed
        self.attached_body = attached_body

        self.intrinsic = intrinsic if intrinsic is not None else np.eye(3)
        # extrinsic should be a 4x4 matrix that maps from the world frame to camera frame
        self.extrinsic = extrinsic if extrinsic is not None else np.eye(4)

        self.depth_factor = depth_factor

    def set_intrinsic(self, intrinsic: np.ndarray):
        """
        Set the intrinsic matrix (3x3).
        """
        self.intrinsic = intrinsic.copy()

    def set_extrinsic(self, extrinsic: np.ndarray):
        """
        Set the extrinsic matrix (4x4).
        """
        self.extrinsic = extrinsic.copy()

    def set_img_dim(self, height: int, width: int):
        """
        Set image dimensions.
        """
        self.height = height
        self.width = width

    def set_depth_factor(self, depth_factor: float):
        """
        Set the depth factor.
        """
        self.depth_factor = depth_factor

    def set_fixed_flag(self, fixed: bool):
        """
        Set the fixed flag.
        """
        self.fixed = fixed

    def set_attached_body(self, attached_body: str):
        """
        Set the attached body.
        """
        self.attached_body = attached_body

    def set_data(
        self,
        timestamp: float,
        height: int,
        width: int,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        fixed: bool,
        attached_body: str,
        depth_factor: float,
    ):
        """
        Set full camera info data.
        """
        self.timestamp = timestamp
        self.height = height
        self.width = width
        self.intrinsic = intrinsic.copy()
        self.extrinsic = extrinsic.copy()
        self.fixed = fixed
        self.attached_body = attached_body
        self.depth_factor = depth_factor

    def get_intrinsic(self) -> np.ndarray:
        """
        Get the intrinsic matrix.
        """
        return self.intrinsic.copy()

    def get_extrinsic(self) -> np.ndarray:
        """
        Get the extrinsic matrix.
        """
        return self.extrinsic.copy()

    def get_img_dim(self):
        """
        Get image dimensions.
        """
        return self.height, self.width

    def get_depth_factor(self) -> float:
        """
        Get the depth factor.
        """
        return self.depth_factor

    def get_fixed_flag(self) -> bool:
        """
        Get the fixed flag.
        """
        return self.fixed

    def get_attached_body(self) -> str:
        """
        Get the attached body.
        """
        return self.attached_body

    def get_data(self):
        """
        Get full camera info data.
        """
        return (
            self.timestamp,
            self.height,
            self.width,
            self.intrinsic.copy(),
            self.extrinsic.copy(),
            self.fixed,
            self.attached_body,
            self.depth_factor,
        )
