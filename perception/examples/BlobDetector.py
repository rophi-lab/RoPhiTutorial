import cv2
import numpy as np

import time
import copy

from perception.BasePerception import BasePerception
from data_type.basic_types.Point3DData import Point3DData
from data_type.basic_types.CameraInfoData import CameraInfoData
from utils.visualization.depth import normalize_depth_image
from utils.perception.camera import pixel_to_camera_with_K, convert_pixel_to_world
from utils.mujoco.camera import (
    get_camera_extrinsics,
    get_camera_intrinsics,
)
from utils.lie.se3 import invSE3


class BlobDetector(BasePerception):
    """
    Perception module that detects spherical blobs using OpenCV.
    NOTE: This module is designed to work with BrlArmGoalEnv.
    """

    def __init__(self, config):
        super().__init__(config)

        self._rgbd_channel = config["sub_manager"]["rgbd_channel"]
        self._point_pub_channel = config["pub_manager"]["point_channel"]

        self._visualization_for_debug = config["visualization_for_debug"]

        self._detector = cv2.SimpleBlobDetector_create()

        self._cam_intrinsic = np.eye(3)
        self._cam2world = np.eye(4)
        self._prev_t = time.time()
        # Intrinsic camera parameter
        # Upper camera (mujoco)
        # TODO: Add these in a config file
        # NOTE: Simulation camera settings should
        # be tunned to match the real world inrinsic later.
        # fovy_rad = np.deg2rad(45)
        # width = 640
        # height = 480
        # fy = 0.5 * height / np.tan(0.5 * fovy_rad)
        # fx = -fy  # in mujoco camera frame, x is pointing to the right
        # self._cam_intrinsic = np.array(
        #     [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]]
        # )

        # # Extrinsic camera parameter
        # self._cam2world = np.array(
        #     [
        #         [0, 0.7071, -0.7071, -2],
        #         [-1, 0, 0.0, 0],
        #         [0, 0.7071, 0.7071, 2],
        #         [0, 0, 0, 1],
        #     ]
        # )
        # self._cam2world = np.eye(4)

    def initialize(self):
        # wait for intrinsic and extrinsic camera parameters
        cam_param_initialized = False
        while not cam_param_initialized:
            if not self.extr_sub_que_dict[self._rgbd_channel + "_info"].empty():
                cam_info = self.extr_sub_que_dict[self._rgbd_channel + "_info"].get()
                if isinstance(cam_info, CameraInfoData):
                    self._cam_intrinsic = cam_info.get_intrinsic()
                    self._cam2world = invSE3(cam_info.get_extrinsic())
                    print(self._cam2world)
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")

    def _process(self):
        """
        Perform blob detection and publish the result.
        """
        # check and extract data from queue
        if not self.extr_sub_que_dict[self._rgbd_channel].empty():
            # pop out the rgbd data for real time tracking
            while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

            rgb_image = rgbd_data.get_rgb_image()
            depth_image = rgbd_data.get_depth_image()
            # print(self._prev_t - time.time())
            self._prev_t = time.time()
            # extract only the red part of the image, and set the rest to white
            gray = self._red_similarity_grayscale(rgb_image)

            # Detect blobs
            best_blob = None
            keypoints = self._detector.detect(gray)

            # find the largest blob
            for kp in keypoints:
                max_size = 0
                for kp in keypoints:
                    if kp.size > max_size:
                        max_size = kp.size
                        best_blob = kp
                pixel = best_blob.pt

            if self._visualization_for_debug:
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                depth_display = normalize_depth_image(depth_image)

                bgr_image_copy = copy.deepcopy(bgr_image)
                if best_blob is not None:
                    # Draw the keypoints on the image
                    bgr_image_copy = cv2.drawKeypoints(
                        bgr_image_copy,
                        keypoints,
                        None,
                        (0, 255, 0),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                    )
                    # Draw the largest blob
                    cv2.circle(
                        bgr_image_copy,
                        (int(pixel[0]), int(pixel[1])),
                        int(best_blob.size / 2),
                        (255, 0, 0),
                        2,
                    )

                cv2.imshow(f"image ({self._rgbd_channel})", bgr_image)
                cv2.imshow(f"detection ({self._rgbd_channel})", bgr_image_copy)
                cv2.imshow(f"depth ({self._rgbd_channel})", depth_display)

            # if the blob is found
            if best_blob is not None:
                # compute the 3d coordinate of the detected blob surface in the world frame
                blob_position = convert_pixel_to_world(
                    pixel,
                    depth_image,
                    self._cam_intrinsic,
                    self._cam2world,
                    inverse_z_direction=False,
                )

                print("Blob position in world frame: ", blob_position)

                # put the point into the Point3DData
                out_pt = Point3DData(num_points=1)
                out_pt.set_time(rgbd_data.get_time())
                out_pt.set_position(blob_position.reshape(1, 3))

                # Publish result (can be original + keypoints or just keypoints)
                if self.percep_pub_que_dict[self._point_pub_channel]:
                    self.percep_pub_que_dict[self._point_pub_channel].put(out_pt)

            if self._visualization_for_debug:
                cv2.waitKey(1)

    def _red_similarity_grayscale(self, rgb_image):
        """
        Computes a grayscale image where intensity reflects how close each pixel is to red.
        Darker means more red. Input must be in RGB format.
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].astype(np.float32)  # Hue channel in [0, 180]

        # Red hue is near 0 or near 180 → wrap around distance
        dist_to_red = np.minimum(np.abs(hue - 0), np.abs(hue - 180))

        # Optionally emphasize saturation (exclude grayish pixels)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        dist_to_red = dist_to_red / 90.0  # Normalize distance to [0, 1]

        dist_to_red[dist_to_red > 0.3] = 1  # Threshold to ignore non-red colors

        similarity = 1.0 - dist_to_red  # Inverted: closer to red = higher similarity
        similarity = similarity * sat  # Saturation-weighted similarity

        grayscale = (1.0 - similarity) * 255.0  # Red = dark, non-red = bright
        return grayscale.astype(np.uint8)
