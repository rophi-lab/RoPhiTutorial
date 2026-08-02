#!/usr/bin/env python3
"""
AprilTag Detection Script

This script subscribes to RGB images, detects AprilTags with a specific ID,
computes the 3D pose of the tag, and publishes it using NamedVecList format.
"""

import numpy as np
import cv2
import time
import logging
from typing import Optional, Tuple
from omegaconf import DictConfig, OmegaConf

# Import required modules from the codebase
from perception.BasePerception import BasePerception
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.NamedVecListData import NamedVecListData
from utils.lie.se3 import invSE3
from utils.filtering.SE3LowPassFilter import SE3LowPassFilter

try:
    from pyapriltags import Detector
except ImportError:
    raise ImportError(
        "AprilTag library is required. Please install it with 'pip install apriltag-python'"
    )


class AprilTagDetector(BasePerception):
    """
    AprilTag detection and pose estimation module.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        # AprilTag detection parameters
        self._target_tag_id = config.get("target_tag_id", 0)
        self._tag_size = config.get("tag_size", 0.05075)  # meters
        self._detector = Detector(
            families="tag25h9",
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=1,
        )

        # Camera parameters
        self._cam_intrinsic: Optional[np.ndarray] = None
        self._cam_intrinsic_vector: Optional[np.ndarray] = None
        self._cam2world: Optional[np.ndarray] = None
        self._depth_factor: Optional[float] = None

        # Pose tracking
        self._pose: Optional[np.ndarray] = None
        self._pose_world: Optional[np.ndarray] = None

        # SE3 Low Pass Filter
        self._use_filter = config.get("pose_filter", {}).get("use_filter", True)
        if self._use_filter:
            filter_alpha = config.get("pose_filter", {}).get("alpha", 0.1)
            filter_window_size = config.get("pose_filter", {}).get("window_size", 0)

            print(f"SE3 Low Pass Filter initialized with alpha={filter_alpha}")
            print(
                f"SE3 Low Pass Filter initialized with window_size={filter_window_size}"
            )
            self._pose_filter: Optional[SE3LowPassFilter] = SE3LowPassFilter(
                alpha=filter_alpha, window_size=filter_window_size
            )
        else:
            self._pose_filter: Optional[SE3LowPassFilter] = None

        # Communication channels
        self._rgbd_channel = config.get("sub_manager", {}).get("rgbd_channel", "d455_1")
        self._object_name = config.get("object_name", f"apriltag_{self._target_tag_id}")
        self._publisher_name = config.get("pub_manager", {}).get("name", "named_vec")

        if self._publisher_name == "named_vec":
            self._pose_pub_channel = config.get("pub_manager", {}).get(
                "named_vec_list_channel", "apriltag_pose"
            )

        # Visualization
        self._visualize = config.get("visualize", False)
        self._window_name = f"AprilTag Detection (ID: {self._target_tag_id})"

        # Debug
        self._debug = config.get("debug", False)

    def initialize(self):
        """
        Initialize camera parameters and visualization.
        """
        if self._visualize:
            self._init_gui()

        # Wait for camera intrinsic and extrinsic parameters
        cam_param_initialized = False
        while not cam_param_initialized:
            if not self.extr_sub_que_dict[self._rgbd_channel + "_info"].empty():
                cam_info = self.extr_sub_que_dict[self._rgbd_channel + "_info"].get()
                if isinstance(cam_info, CameraInfoData):
                    self._cam_intrinsic = cam_info.get_intrinsic()
                    self._cam_intrinsic_vector = np.array(
                        [
                            self._cam_intrinsic[0, 0],
                            self._cam_intrinsic[1, 1],
                            self._cam_intrinsic[0, 2],
                            self._cam_intrinsic[1, 2],
                        ]
                    )
                    self._cam2world = invSE3(cam_info.get_extrinsic())
                    self._depth_factor = cam_info.get_depth_factor()
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")
                    if self._debug:
                        print(f"Camera intrinsic: {self._cam_intrinsic}")
                        print(f"Camera to world transform: {self._cam2world}")

    def stop(self):
        """
        Stop the detector and clean up.
        """
        if self._visualize:
            cv2.destroyAllWindows()
        super().stop()

    def _process(self):
        """
        Process RGB images to detect AprilTags and estimate pose.
        """
        if not self.extr_sub_que_dict[self._rgbd_channel].empty():
            # Get the latest RGB data
            # rgb_data = self.extr_sub_que_dict[self._rgbd_channel].get()

            # Clear the queue to get the most recent data
            while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                rgb_data = self.extr_sub_que_dict[self._rgbd_channel].get()
            
            
            rgb_image = rgb_data.get_rgb_image()

            if rgb_image is None:
                logging.warning("[AprilTagDetector] RGB image is None.")
                return

            # Detect AprilTags
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            results = self._detector.detect(gray_image,
                                            estimate_tag_pose=True,
                                            camera_params=self._cam_intrinsic_vector,
                                            tag_size=self._tag_size,
                                            )

            # print(results.__len__(), "tags detected")

            # Find target tag
            target_result = None
            filtered_pose = None
            for result in results:
                if result.tag_id == self._target_tag_id:
                    target_result = result
                    break

            if target_result is not None and self._cam_intrinsic is not None:
                # Estimate pose
                # pose, e0, e1 = self._detector.detection_pose(
                #     target_result,
                #     (
                #         self._cam_intrinsic[0, 0],
                #         self._cam_intrinsic[1, 1],
                #         self._cam_intrinsic[0, 2],
                #         self._cam_intrinsic[1, 2],
                #     ),
                #     self._tag_size,
                # )

                pose_R = target_result.pose_R
                pose_t = target_result.pose_t
                
                pose = np.eye(4)
                pose[0:3,0:3] = pose_R
                pose[0:3,3] = pose_t.reshape(3,)

                
                if pose is not None:
                    # Apply SE3 low pass filter if enabled
                    if self._use_filter and self._pose_filter is not None:
                        filtered_pose = self._pose_filter.update(pose)
                    else:
                        filtered_pose = pose

                    # Transform to world coordinates
                    self._pose_world = self._cam2world @ filtered_pose

                    # Publish the pose in named_vec format
                    if self._publisher_name == "named_vec":
                        pose_out = np.concatenate(
                            [
                                self._pose_world[:3, 3],
                                self._pose_world[:3, :3].flatten(),
                            ],
                            axis=0,
                        ).reshape(1, 12)

                        out_data = NamedVecListData(
                            num_vecs=1, vec_dim=12, name=self._object_name
                        )
                        out_data.set_data(
                            rgb_data.get_time(), [self._object_name], pose_out
                        )

                        if self.percep_pub_que_dict[self._pose_pub_channel]:
                            self.percep_pub_que_dict[self._pose_pub_channel].put(
                                out_data
                            )

                    if self._debug:
                        print(
                            f"Tag {self._target_tag_id} detected at position: {self._pose_world[:3, 3]}"
                        )

                    # Visualize if enabled
            if self._visualize:
                # cv2.imshow(self._window_name, rgb_image)
                # cv2.waitKey(1)
                self._visualize_detection(
                    rgb_image, target_result, filtered_pose
                )

            # elif self._debug:
            # print(f"Target tag {self._target_tag_id} not found in image.")

    def _init_gui(self):
        """
        Initialize the OpenCV window for visualization.
        """
        cv2.namedWindow(self._window_name)
        cv2.resizeWindow(self._window_name, 800, 600)

    def _visualize_detection(self, rgb_image: np.ndarray, result, pose: np.ndarray):
        """
        Visualize the detected AprilTag with pose axes.

        Args:
            rgb_image: RGB image
            result: AprilTag detection result
            pose: 4x4 pose matrix
        """
        # Convert to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if pose is not None:
            # Draw tag border
            corners = np.array(result.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                bgr_image, [corners], isClosed=True, color=(0, 255, 0), thickness=2
            )

            # Draw tag center
            center = tuple(np.round(result.center).astype(int))
            cv2.circle(bgr_image, center, 4, (0, 0, 255), -1)

            # Draw tag ID
            cv2.putText(
                bgr_image,
                f"ID: {result.tag_id}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            # Draw pose axes if camera intrinsics are available
            if self._cam_intrinsic is not None:
                # Project 3D axes to 2D
                axis_length = self._tag_size / 2
                axes_3d = np.float32(
                    [
                        [0, 0, 0],
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length],
                    ]
                ).reshape(-1, 3)

                # Extract rotation and translation from pose
                rvec, _ = cv2.Rodrigues(pose[:3, :3])
                tvec = pose[:3, 3]

                # Project points
                camera_matrix = np.array(
                    [
                        [self._cam_intrinsic[0, 0], 0, self._cam_intrinsic[0, 2]],
                        [0, self._cam_intrinsic[1, 1], self._cam_intrinsic[1, 2]],
                        [0, 0, 1],
                    ]
                )
                dist_coeffs = np.zeros(5)

                imgpts, _ = cv2.projectPoints(
                    axes_3d, rvec, tvec, camera_matrix, dist_coeffs
                )
                imgpts = imgpts.astype(int)

                # Draw axes
                origin = tuple(imgpts[0].ravel())
                cv2.line(
                    bgr_image, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 2
                )  # X axis (red)
                cv2.line(
                    bgr_image, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 2
                )  # Y axis (green)
                cv2.line(
                    bgr_image, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 2
                )  # Z axis (blue)

                # Add pose information
                pos_text = f"Pos: ({pose[0, 3]:.3f}, {pose[1, 3]:.3f}, {pose[2, 3]:.3f})"
                cv2.putText(
                    bgr_image,
                    pos_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
        
        # Show image
        cv2.imshow(self._window_name, bgr_image)
        cv2.waitKey(1)


def main():
    """
    Main function to run the AprilTag detector.
    """
    # Default configuration
    config = OmegaConf.create(
        {
            "target_tag_id": 0,
            "tag_size": 0.05075,  # meters
            "visualize": True,
            "debug": True,
            "pose_filter": {"use_filter": False, "alpha": 0.1, "window_size": 0},
            "sub_manager": {"rgbd_channel": "d455_1"},
            "pub_manager": {
                "name": "named_vec",
                "named_vec_list_channel": "apriltag_pose",
            },
            "object_name": "hw_apriltag",
        }
    )

    # Create and run the detector
    detector = AprilTagDetector(config)

    try:
        detector.initialize()
        detector.start()
    except KeyboardInterrupt:
        print("\n[AprilTagDetector] Interrupted by user.")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
