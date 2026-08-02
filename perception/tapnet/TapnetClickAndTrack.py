import os
import numpy as np
import time
import cv2
import lcm
import jax
from typing import Optional
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from tapnet.torch import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

import torch
import torch.nn.functional as F
import cv2
import copy


from data_type.basic_types.Point3DData import Point3DData
from data_type.basic_types.CameraInfoData import CameraInfoData
from perception.BasePerception import BasePerception
from utils.perception.camera import convert_pixel_to_world
from utils.lie.se3 import invSE3


class TapnetClickAndTrack(BasePerception):
    def __init__(self, config):
        super().__init__(config)
        self.name = "TapnetClickAndTrack"

        self._resize_height = config.get("resize_height", 256)
        self._resize_width = config.get("resize_width", 256)

        self._update_freq = config.get("update_freq", 10.0)
        self._update_dt = 1 / self._update_freq
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_name = config.get(
            "checkpoint_name", "causal_bootstapir_checkpoint.pt"
        )
        # get current path
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), "checkpoints/", checkpoint_name
        )

        # TAPIR model
        self._model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self._model.load_state_dict(torch.load(checkpoint_path))
        self._model = self._model.to(self._device)
        self._model = self._model.eval()
        torch.set_grad_enabled(False)

        # model related variables
        self._query_points = None
        self._query_features = None
        self._causal_state = None
        self._is_first_frame = True
        # N number of points from previous tracking results (N,2)
        self._last_track_pts: Optional[torch.Tensor] = None
        # N number of uncertainties from previous tracking results (N,1)
        # value is normalized to [0,1] with sigmoid
        self._last_track_uncertainty = None
        # the pixel location of the last reliable tracked point
        # that is converted to the world coordinate and send
        # to the controller
        self._last_reliable_pt: Optional[np.ndarray] = None
        self._uncertainty_thres = config.get("uncertainty_threshold", 0.3)

        # flag to use multiple points for tracking one object or not
        self._use_multi_points = config.get("use_multi_points", False)
        self._num_sample_points = config.get("num_sample_points", 10)
        self._sample_box_size = int(config.get("sample_box_size", 20) / 2)
        self._vis_colors = self._get_n_colors(self._num_sample_points + 1)

        self._rgbd_channel = config["sub_manager"]["rgbd_channel"]
        self._point_pub_channel = config["pub_manager"]["point_channel"]

        # self._visualization_for_debug = config["visualization_for_debug"]

        # camera related info will be updated after camera info is received
        self._cam_intrinsic = np.eye(3)
        self._cam2world = np.eye(4)
        self._depth_factor = 1.0
        self._img_height = 480
        self._img_width = 640
        self._vis_point_pixels = None

        self._first_pt_clicked = False
        self._last_clicked_pt = None
        self._last_update_t = time.time()

    def initialize(self):
        # initialize gui
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
                    self._img_height, self._img_width = cam_info.get_img_dim()
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    def _process(self):
        """
        Display the frame and return clicked pixel coordinate if any.
        This is designed to be called in a while loop.
        """
        if not self.extr_sub_que_dict[self._rgbd_channel].empty():

            # pop out the rgbd data for real time tracking
            while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

            rgb_image = rgbd_data.get_rgb_image()
            depth_image = rgbd_data.get_depth_image()

            if time.time() - self._last_update_t < self._update_dt:
                return
            else:
                self._last_update_t = time.time()

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            # print(self._device)
            if self._last_track_pts is not None:
                # print("last track pixels: ", self._last_track_pts)
                for i in range(self._last_track_pts.shape[0]):
                    cv2.circle(
                        bgr_image,
                        self._last_track_pts[i, :].numpy().astype(int).reshape(2),
                        radius=5,
                        color=self._vis_colors[i],
                        thickness=-1,
                    )

                cv2.circle(
                    bgr_image,
                    self._last_reliable_pt.astype(int).reshape(2),
                    radius=10,
                    color=(0, 0, 255),
                    thickness=-1,
                )

            cv2.imshow(self._window_name, bgr_image)
            # cv2.imshow("Depth Image", depth_image)

            key = cv2.waitKey(1)  # Add key check if needed (e.g. for exit)

            # if we get a new clicked piont,
            # we need re-initialize the model
            if self.clicked_point is not None:
                self._first_pt_clicked = True
                self._last_clicked_pt = np.array(self.clicked_point).reshape(1, 2)
                self._last_reliable_pt = np.array(self.clicked_point).reshape(1, 2)
                self._last_track_pts = np.array(self.clicked_point).reshape(1, 2)
                self._is_first_frame = True

            # we don't do anything until we receive a clicked point
            if not self._first_pt_clicked:
                return
            else:
                rgb_resize = cv2.resize(
                    rgb_image, (self._resize_width, self._resize_height)
                )
                frame = torch.tensor(rgb_resize).to(self._device)

                if self._is_first_frame:
                    sampled_points = self._sample_query_points(self._last_clicked_pt)

                    self._query_points = self._convert_select_points_to_query_points(
                        0, np.array(sampled_points)
                    )

                    self._query_points = torch.tensor(self._query_points).to(
                        self._device
                    )
                    # print("query points: ", self._query_points)

                    # Initialize query features
                    self._query_features = self._online_model_init(
                        self._model,
                        frame.unsqueeze(0).unsqueeze(0),
                        self._query_points[None],
                    )
                    self._causal_state = self._model.construct_initial_causal_state(
                        self._query_points.shape[0],
                        len(self._query_features.resolutions) - 1,
                    )

                    with torch.no_grad():
                        for i in range(len(self._causal_state)):
                            for k, v in self._causal_state[i].items():
                                self._causal_state[i][k] = v.to(self._device)

                    self._is_first_frame = False
                    self.clicked_point = None  # Reset after read

                with torch.no_grad():
                    # Predict trajectories and occlusions
                    tracks, uncertainty, visibles, self._causal_state = (
                        self._online_model_predict(
                            self._model,
                            frame.unsqueeze(0).unsqueeze(0),
                            self._query_features,
                            self._causal_state,
                        )
                    )
                    self._last_track_uncertainty = uncertainty.cpu()
                    self._last_track_pts = (
                        transforms.convert_grid_coordinates(
                            copy.deepcopy(tracks).cpu(),
                            (self._resize_width, self._resize_height),
                            (self._img_width, self._img_height),
                        )
                        .squeeze(0)
                        .squeeze(0)
                        .squeeze(0)
                    )

            if self._last_track_pts is not None:
                pt_filtered = self._last_track_pts[
                    (self._last_track_uncertainty < self._uncertainty_thres).squeeze(0)
                ]
                # pt = self._last_track_pts[0, :].reshape(2)
                if pt_filtered.shape[0] == 0:
                    pt = self._last_reliable_pt
                else:
                    # print("filtered pixels: ", pt_filtered)
                    # print(pt_filtered.shape)
                    pt = torch.mean(pt_filtered, axis=0)
                    self._last_reliable_pt = pt.numpy()
                # self.clicked_point = None  # Reset after read
                # print("tracked pixel: ", pt.shape)
                # print("tracked pixel: ", pt)
                # Convert pixel coordinates to world coordinates
                point_world = convert_pixel_to_world(
                    pt,
                    depth_image,
                    self._cam_intrinsic,
                    self._cam2world,
                    depth_factor=self._depth_factor,
                    inverse_z_direction=False,
                )
                if point_world is None:
                    print("Invalid depth value at pixel location.")
                else:

                    # Create Point3DData object and publish it
                    out_3d_pt = Point3DData(num_points=1)
                    out_3d_pt.set_time(rgbd_data.get_time())
                    out_3d_pt.set_position(point_world.reshape(1, 3))

                    # Publish result (can be original + keypoints or just keypoints)
                    if self.percep_pub_que_dict[self._point_pub_channel]:
                        self.percep_pub_que_dict[self._point_pub_channel].put(out_3d_pt)
                        print(f"Published 3D point: {point_world}")

    def _preprocess_frames(self, frames):
        """Preprocess frames to model inputs.

        Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8

        Returns:
        frames: [num_frames, height, width, 3], [-1, 1], np.float32
        """
        frames = frames.float()
        frames = frames / 255 * 2 - 1
        return frames

    def _sample_query_points(self, click_point):
        if not self._use_multi_points:
            return click_point
        else:
            # Sample random points around the clicked point

            points = self._sample_random_points_in_box_around_pt(
                click_point, self._num_sample_points, box_size=self._sample_box_size
            )
            points = np.concatenate((points, click_point), axis=0)
            return points

    def _convert_select_points_to_query_points(self, frame_id, points):
        """Convert select points to query points.

        Args:
            points: [num_points, 2], [t, y, x]

        Returns:
            query_points: [num_points, 3], [t, y, x]
        """
        points = np.stack(points)
        query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
        query_points[:, 0] = frame_id
        query_points[:, 1] = points[:, 1] / self._img_height * self._resize_height
        query_points[:, 2] = points[:, 0] / self._img_width * self._resize_width
        return query_points

    def _sample_random_points_in_box_around_pt(
        self,
        pt,
        num_points,
        box_size=10,
    ):
        """
        Sample unique random points in a box around the clicked point.

        param[in] pt: [1,2], (x, y)
        param[in] num_points: number of points to sample
        param[in] box_size: size of the box to sample points from. Unit: pixels
        """

        # Create a grid of all possible points within the box
        x_range = np.arange(pt[0, 0] - box_size, pt[0, 0] + box_size + 1)
        y_range = np.arange(pt[0, 1] - box_size, pt[0, 1] + box_size + 1)
        mesh_x, mesh_y = np.meshgrid(x_range, y_range)
        all_points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)

        # Remove the center point if needed (optional)
        all_points = all_points[~np.all(all_points == pt, axis=1)]

        # Shuffle and select unique points
        np.random.shuffle(all_points)
        num_points = min(num_points, len(all_points))
        points = all_points[:num_points]

        return points  # shape: [num_points, 2]

    def _postprocess_occlusions(self, occlusions, expected_dist):
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
        return visibles

    def _online_model_init(self, model, frames, query_points):
        """Initialize query features for the query points."""
        frames = self._preprocess_frames(frames)
        feature_grids = model.get_feature_grids(frames, is_training=False)

        query_features = model.get_query_features(
            frames,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        return query_features

    def _online_model_predict(self, model, frames, query_features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        # t0 = time.time()
        frames = self._preprocess_frames(frames)
        # print("-----------------------------")
        # print("preprocess frame time:", time.time() - t0)
        # t = time.time()
        feature_grids = model.get_feature_grids(frames, is_training=False)
        # print("get_feature_grids time:", time.time() - t)
        # t = time.time()
        trajectories = model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=query_features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        # print("estimate_trajectories time:", time.time() - t)
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        # print("del time: ", time.time() - t)
        # Take only the predictions for the final resolution.
        # For running on higher resolution, it's typically better to average across
        # resolutions.
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        expected_distance = trajectories["expected_dist"][-1]
        uncertainty = copy.deepcopy(F.sigmoid(expected_distance))
        # print("uncertainty: ", F.sigmoid(uncertainty))
        # t = time.time()
        visibles = self._postprocess_occlusions(occlusions, expected_distance)
        # print("postprocess_occlusions time:", time.time() - t)
        # print("t end: ", time.time() - t0)
        return tracks, uncertainty, visibles, causal_context

    def ransac_filter_pixels(
        self,
        points,
        uncertainty,
        num_iterations=100,
        threshold=5.0,
        min_inliers_ratio=0.5,
    ):
        """
        RANSAC-based outlier removal for 2D pixel tracks.

        Args:
            pixels: (N, 2) numpy array of tracked pixel coordinates
            uncertainty: (N, 1) numpy array of uncertainty values
            num_iterations: number of RANSAC iterations
            threshold: pixel distance threshold (can be scaled by uncertainty)
            min_inliers_ratio: minimum fraction of inliers to accept a model

        Returns:
            inlier_mask: Boolean array of shape (N,) where True indicates an inlier
        """
        N = points.shape[0]
        best_inliers = np.zeros(N, dtype=bool)
        max_inliers = 0

        for _ in range(num_iterations):
            # Randomly sample a single point as the "hypothesis"
            idx = np.random.choice(N)
            ref_pixel = points[idx]

            # Compute distances from all points to the reference point
            dists = np.linalg.norm(points - ref_pixel, axis=1)

            # Normalize threshold by uncertainty
            scaled_threshold = threshold * (
                uncertainty.squeeze() + 1e-6
            )  # Avoid div by 0

            inliers = dists < scaled_threshold

            if np.sum(inliers) > max_inliers:
                max_inliers = np.sum(inliers)
                best_inliers = inliers

            # Early exit if good enough
            if max_inliers > min_inliers_ratio * N:
                break

        return best_inliers

    def _init_gui(self):
        """
        Initializes the OpenCV window and mouse callback once.
        Should be called once in __init__ or before _process.
        """
        self.clicked_point = None
        self._window_name = "Click on Image"
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print(f"Clicked pixel: ({x}, {y})")

    # Generate N distinct colors from a colormap

    def _get_n_colors(self, n):
        cmap = plt.get_cmap("hsv")  # or 'tab20', 'jet', etc.
        colors = [tuple(int(c * 255) for c in cmap(i / n)[:3]) for i in range(n)]
        return colors
