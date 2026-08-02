"""
FoundationPosePerception module for pose tracking from RGBD data via LCM, using FoundationPose.
"""

from collections import defaultdict
from queue import Queue, Empty
import numpy as np
import os
import logging
from typing import Optional

# SAM is required for the first frame pose estimation
from segment_anything import sam_model_registry, SamPredictor

from perception.BasePerception import BasePerception
from data_type.basic_types.CameraInfoData import CameraInfoData
from utils.visualization.bounding_box import draw_posed_3d_box
from utils.lie.se3 import invSE3


try:
    import cv2  # OpenCV is required for visualization
except ImportError as e:
    raise ImportError(
        "OpenCV (cv2) is required for FoundationPoseTracking visualization. Please install it with 'pip install opencv-python'."
    ) from e


class SAM(BasePerception):
    """
    Perception module for pose tracking from RGBD data via LCM using FoundationPose.
    """

    def __init__(self, config):
        super().__init__(config)

        # SAM related
        self._sam_model_type = config.get("sam_model_type", "vit_h")
        self._sam_ckpt = config.get("sam_ckpt", "sam_vit_h_4b8939.pth")
        self._sam_model = sam_model_registry[self._sam_model_type](
            checkpoint="perception/sam/checkpoints/" + self._sam_ckpt
        )
        self._sam_predictor = SamPredictor(self._sam_model)
        self._visualize_sam = config.get("visualize_sam", False)

        # camera parameters
        self._cam_intrinsic: Optional[np.ndarray] = None
        self._cam2world: Optional[np.ndarray] = None
        self._depth_factor: Optional[float] = None

        self._initialized = False

        self._rgbd_channel = config.get("sub_manager", "single_rgbd").get(
            "rgbd_channel", "d455_1"
        )

        self._visualize = config.get("visualize", False)
        self._clicked_point = None

        self._save_results = config.get("save_results", False)
        self._save_dir = config.get("save_dir", "data/sam_ycb_output")
        os.makedirs(self._save_dir, exist_ok=True)
        self._save_idx = 0
        self._last_mask = None
        self._last_rgb = None
        self._last_depth = None

        self._window_name = "SAM"
        # os.makedirs(self._debug_dir, exist_ok=True)
        # cv2.namedWindow(self._window_name)

    def initialize(self):
        """
        Load mesh/model and initialize the pose estimator.
        """

        self._init_gui()

        # wait for intrinsic and extrinsic camera parameters
        cam_param_initialized = False
        while not cam_param_initialized:
            if not self.extr_sub_que_dict[self._rgbd_channel + "_info"].empty():
                cam_info = self.extr_sub_que_dict[self._rgbd_channel + "_info"].get()
                if isinstance(cam_info, CameraInfoData):
                    self._cam_intrinsic = cam_info.get_intrinsic()
                    self._cam2world = invSE3(cam_info.get_extrinsic())
                    self._depth_factor = cam_info.get_depth_factor()
                    # print(self._cam2world)
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    def _process(self):
        """
        This method must be overridden to match the abstract base class signature and always raise NotImplementedError.
        Use process_once() for the actual logic.
        """
        if not self.extr_sub_que_dict[self._rgbd_channel].empty():
            rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()
            rgb_image = rgbd_data.get_rgb_image()
            depth_image = rgbd_data.get_depth_image()

            self._vis_image(rgb_image.copy(), self._window_name)

            if self._clicked_point is not None:
                # visualize the clicked point
                rgb_vis = rgb_image.copy()
                cv2.circle(
                    rgb_vis,
                    self._clicked_point,
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,
                )
                self._vis_image(rgb_vis, self._window_name)

                # set the image to SAM
                self._sam_predictor.set_image(rgb_vis)

                # Get the mask for the object
                input_point = np.array(
                    [[self._clicked_point[0], self._clicked_point[1]]]
                )
                input_label = np.array([1])
                masks, scores, logits = self._sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                # take the last mask (the largest one)
                mask = masks[-1]
                self._last_mask = mask.astype(np.uint8)
                self._last_rgb = rgb_image.copy()
                self._last_depth = depth_image.copy()

                if self._save_ycb_sample:
                    # Save in YCB format
                    self._save_ycb_sample()

                if self._visualize_sam:
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    overlay = bgr_image.copy()
                    overlay[mask > 0] = (0, 255, 0)  # Green overlay
                    vis = cv2.addWeighted(bgr_image, 0.7, overlay, 0.3, 0)
                    cv2.circle(
                        vis,
                        self._clicked_point,
                        radius=5,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
                    cv2.imshow("SAM Segmentation", vis)

                # After saving, reset for next click
                self._clicked_point = None

            # else:
            #     logging.warning("[FoundationPosePerception] Color image is None.")
        # raise NotImplementedError("Subclasses should implement this method.")

    def _init_gui(self):
        """
        Initializes the OpenCV window and mouse callback once.
        Should be called once in __init__ or before _process.
        """
        self._clicked_point = None
        self._window_name = "FoundationPoseTracking"
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked_point = (x, y)
            print(f"Clicked pixel: ({x}, {y})")
            self._initialized = False  # Allow processing for each click

    def _vis_image(self, image, window_name="robot_view"):
        # Convert RGB to BGR for OpenCV
        if image.shape[-1] == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image
        cv2.imshow(window_name, bgr_image)
        cv2.waitKey(1)

    def _save_ycb_sample(self):
        """
        Save the current RGB, depth, and mask in YCB dataset format.
        """
        if (
            self._last_rgb is None
            or self._last_depth is None
            or self._last_mask is None
        ):
            print("[SAM] Missing data for saving YCB sample.")
            return

        idx = self._save_idx

        # Save color image (YCB: color/xxxxxx.png)
        color_dir = os.path.join(self._save_dir, "rgb")
        os.makedirs(color_dir, exist_ok=True)
        color_path = os.path.join(color_dir, f"{idx:06d}.png")
        rgb_bgr = cv2.cvtColor(self._last_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(color_path, rgb_bgr)

        # Save depth image (YCB: depth/xxxxxx.png)
        depth_dir = os.path.join(self._save_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        depth_path = os.path.join(depth_dir, f"{idx:06d}.png")
        cv2.imwrite(depth_path, self._last_depth)

        # Save mask (YCB: mask/xxxxxx.png)
        mask_dir = os.path.join(self._save_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"{idx:06d}.png")
        cv2.imwrite(mask_path, (self._last_mask * 255).astype(np.uint8))

        print(f"[SAM] Saved YCB sample to {self._save_dir}")
        self._save_idx += 1
