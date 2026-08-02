"""
FoundationPosePerception module for pose tracking from RGBD data via LCM, using FoundationPose.
"""

from collections import defaultdict
from queue import Queue, Empty
import numpy as np
import os
import logging
from typing import Optional

from sympy import bool_map
import copy

# foundation pose requirements
import torch
import trimesh
import nvdiffrast.torch as dr
from third_party.FoundationPose.Utils import depth2xyzmap, toOpen3dCloud
from third_party.FoundationPose.gsplat_renderer import make_gaussian_tensors
import open3d as o3d

# SAM2 is required for the first frame pose estimation
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_camera_predictor

from perception.BasePerception import BasePerception
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.NamedVecListData import NamedVecListData
from utils.visualization.bounding_box import draw_posed_3d_box, draw_xyz_axis
from utils.lie.se3 import invSE3
from utils.filtering.SE3LowPassFilter import SE3LowPassFilter


from third_party.FoundationPose.estimater import FoundationPose
from third_party.FoundationPose.learning.training.predict_score import ScorePredictor
from third_party.FoundationPose.learning.training.predict_pose_refine import (
    PoseRefinePredictor,
)


try:
    import cv2  # OpenCV is required for visualization
except ImportError as e:
    raise ImportError(
        "OpenCV (cv2) is required for FoundationPoseTracking visualization. Please install it with 'pip install opencv-python'."
    ) from e


def _to_origin_from_neighbor_mesh(splat_path, means):
    """Return (to_origin, extents) canonicalizing the object frame from the mesh
    sitting next to a gaussian-splat file, falling back to the gaussians' own
    oriented bounds when no mesh is found.

    The published object pose ("bb" frame) must match what pose consumers
    (controller bb->cad, MeshGraspVisualizer) reconstruct via
    trimesh.bounds.oriented_bounds(mesh). oriented_bounds() picks principal-axis
    signs arbitrarily, so computing it on the gaussian means can yield a frame
    flipped 180 deg from the mesh's -> flipped rendered mesh. Deriving to_origin
    from the mesh guarantees agreement (valid because the PCA-aligned splats
    share the mesh's native frame).
    """
    splat_dir = os.path.dirname(splat_path)
    for _name in ("textured_mesh.obj", "textured_mesh.ply"):
        mesh_path = os.path.join(splat_dir, _name)
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            return trimesh.bounds.oriented_bounds(mesh)
    logging.warning(
        "[FoundationPose] no textured_mesh.{obj,ply} beside %s; canonicalizing "
        "the bb frame from the gaussian means, which may be flipped 180 deg "
        "relative to the mesh consumers use.",
        splat_path,
    )
    return trimesh.bounds.oriented_bounds(means)


class FoundationPoseTracking(BasePerception):
    """
    Perception module for pose tracking from RGBD data via LCM using FoundationPose.
    """

    def __init__(self, config):
        super().__init__(config)

        # mesh file, or splat_file for a Gaussian-Splat object model (mutually exclusive)
        self._splat_path = config.get("splat_file", None)
        self._is_gaussian = bool(self._splat_path)

        if self._is_gaussian:
            self._mesh_path = None
            self._mesh = None
            self._gaussian_tensors = make_gaussian_tensors(self._splat_path, device="cuda")
            means = self._gaussian_tensors["means"].detach().cpu().numpy()
            # Canonicalize the published (bb) frame from the neighbor mesh, not
            # the gaussians' own oriented bounds (which can be flipped 180 deg
            # by oriented_bounds() axis-sign ambiguity). See helper docstring.
            self._to_origin, extents = _to_origin_from_neighbor_mesh(
                self._splat_path, means
            )
            model_pts, model_normals = means, None
        else:
            self._mesh_path = config.get("mesh_file", None)
            self._mesh = trimesh.load(self._mesh_path)
            self._gaussian_tensors = None
            self._to_origin, extents = trimesh.bounds.oriented_bounds(self._mesh)
            model_pts, model_normals = self._mesh.vertices, self._mesh.vertex_normals

        self._to_origin_inv = np.linalg.inv(self._to_origin)
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        self._pose: Optional[np.ndarray] = None
        self._pose_world: Optional[np.ndarray] = None

        # SE3 Low Pass Filter
        self._use_filter = config.get("pose_filter").get("use_filter", True)
        if self._use_filter:
            filter_alpha = config.get("pose_filter").get("alpha", 0.1)
            filter_window_size = config.get("pose_filter").get("window_size", 0)

            print(f"SE3 Low Pass Filter initialized with alpha={filter_alpha}")
            print(
                f"SE3 Low Pass Filter initialized with window_size={filter_window_size}"
            )
            self._pose_filter: Optional[SE3LowPassFilter] = SE3LowPassFilter(
                alpha=filter_alpha, window_size=filter_window_size
            )
        else:
            self._pose_filter: Optional[SE3LowPassFilter] = None

        self._fix_x_rotation = config.get("pose_filter").get("fix_x_rotation", False)
        self._fix_z_rotation = config.get("pose_filter").get("fix_z_rotation", False)

        # FoundationPose related
        self._est_refine_iter = config.get("est_refine_iter", 5)
        # Register pose-hypothesis grid size (fewer views => faster register).
        self._register_min_n_views = config.get("register_min_n_views", 40)
        self._register_inplane_step = config.get("register_inplane_step", 60)
        self._track_refine_iter = config.get("track_refine_iter", 2)
        self._debug = config.get("debug", 0)
        self._debug_dir = config.get("debug_dir", "./debug")

        # logging / FPS. FoundationPose's Utils.py calls logging.basicConfig on
        # the root logger at import, flooding the console with per-frame INFO;
        # quiet_logs raises the root level to WARNING. fps_report_every prints
        # aggregate tracking FPS every N frames (0 = never).
        self._quiet_logs = config.get("quiet_logs", True)
        self._fps_report_every = config.get("fps_report_every", 30)
        if self._quiet_logs:
            logging.getLogger().setLevel(logging.WARNING)
            for _n in ("FoundationPose", "sam2", "SAM2", "predict_score",
                       "predict_pose_refine"):
                logging.getLogger(_n).setLevel(logging.WARNING)
        self._fps_frame_count = 0
        self._fps_window_start = None

        self._glctx = dr.RasterizeCudaContext()
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()

        # Render/crop padding: crop window radius = mesh_diameter*crop_ratio/2,
        # so larger = more margin around the object in the render+observed crop.
        # Read fresh at predict time, so overriding both nets' cfg takes effect.
        # Released weights train at 1.2; keep modest (≈1.4-1.6) since the object
        # shrinks in the fixed input window and can hurt accuracy. Omit to leave
        # the model default untouched.
        crop_ratio = config.get("crop_ratio", None)
        if crop_ratio is not None:
            self._scorer.cfg["crop_ratio"] = float(crop_ratio)
            self._refiner.cfg["crop_ratio"] = float(crop_ratio)
            logging.warning(
                "[FoundationPose] crop_ratio overridden to %.3f (default 1.2); "
                "larger = more crop padding but can degrade pose accuracy.",
                float(crop_ratio),
            )

        self._estimator = FoundationPose(
            model_pts=model_pts,
            model_normals=model_normals,
            mesh=self._mesh,
            scorer=self._scorer,
            refiner=self._refiner,
            debug_dir=self._debug_dir,
            debug=self._debug,
            glctx=self._glctx,
        )
        self._estimator.make_rotation_grid(
            min_n_views=self._register_min_n_views,
            inplane_step=self._register_inplane_step,
        )

        # SAM2 related (only used for the first frame mask)
        sam2_cfg = config.get("sam2", {})
        self._sam2_config_file = sam2_cfg.get(
            "config_file", "configs/sam2.1/sam2.1_hiera_t.yaml"
        )
        # Prefer a local checkpoint when provided; fall back to HF Hub download.
        self._sam2_checkpoint = sam2_cfg.get("checkpoint", None)
        self._sam2_hf_repo = sam2_cfg.get("hf_repo", "facebook/sam2.1-hiera-tiny")
        self._sam2_hf_filename = sam2_cfg.get("hf_filename", "sam2.1_hiera_tiny.pt")
        self._sam2_obj_id = sam2_cfg.get("obj_id", 1)
        self._sam_predictor = None
        self._visualize_sam = config.get("visualize_sam", False)

        # camera parameters
        self._cam_intrinsic: Optional[np.ndarray] = None
        self._cam2world: Optional[np.ndarray] = None
        self._depth_factor: Optional[float] = None

        self._initialized = False

        self._rgbd_channel = config.get("sub_manager", "single_rgbd").get(
            "rgbd_channel", "d455_1"
        )

        self._object_name = config.get("object_name", "predefined_obj")
        self._publisher_name = config["pub_manager"]["name"]
        if self._publisher_name == "named_vec":
            self._pose_pub_channel = config["pub_manager"]["named_vec_list_channel"]

        self._visualize = config.get("visualize", False)
        self._clicked_point = None

        self._window_name = "FoundationPoseTracking RGB"

    def initialize(self):
        """
        Load mesh/model and initialize the pose estimator.
        """

        self._init_gui()

        logging.info(
            f"initializing SAM2CameraPredictor ({self._sam2_config_file})"
        )
        if self._sam2_checkpoint and os.path.isfile(self._sam2_checkpoint):
            ckpt_path = self._sam2_checkpoint
        else:
            if self._sam2_checkpoint:
                logging.warning(
                    f"sam2.checkpoint not found at {self._sam2_checkpoint}, "
                    f"falling back to HF Hub ({self._sam2_hf_repo}/{self._sam2_hf_filename})"
                )
            ckpt_path = hf_hub_download(
                repo_id=self._sam2_hf_repo, filename=self._sam2_hf_filename
            )
        self._sam_predictor = build_sam2_camera_predictor(
            self._sam2_config_file, ckpt_path, device="cuda", vos_optimized=True
        )
        logging.info(f"SAM2CameraPredictor initialized (ckpt={ckpt_path})")

        # wait for intrinsic and extrinsic camera parameters
        cam_param_initialized = False
        while not cam_param_initialized:
            if not self.extr_sub_que_dict[self._rgbd_channel + "_info"].empty():
                cam_info = self.extr_sub_que_dict[self._rgbd_channel + "_info"].get()
                if isinstance(cam_info, CameraInfoData):
                    # float32: CameraInfoData builds K as float64, but FoundationPose
                    # runs in cuda.FloatTensor. A float64 K makes register() produce
                    # float64 ob_in_cams, which then clashes with the float32 K in
                    # compute_crop_window_tf_batch (K@pts: float != double).
                    self._cam_intrinsic = cam_info.get_intrinsic().astype(np.float32)
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
            # pop out the rgbd data for real time tracking
            while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

            rgb_image = rgbd_data.get_rgb_image()
            depth_image = rgbd_data.get_depth_image()
            depth_image_meter = (
                depth_image.copy().astype(np.float32) / self._depth_factor
            )
            # print(self._depth_factor)
            # FoundationPose require segmentation mask for the first frame
            # We wait for the user input and use SAM to get the mask for the first frame
            if not self._initialized:
                self._vis_image(rgb_image.copy(), self._window_name)
                # Normalize depth image for visualization
                # if np.issubdtype(depth_image.dtype, np.floating):
                #     depth_vis = (depth_image - np.min(depth_image)) / (
                #         np.max(depth_image) - np.min(depth_image) + 1e-5
                #     )
                #     depth_display = (depth_vis * 255).astype(np.uint8)
                # else:
                #     depth_display = cv2.convertScaleAbs(depth_image)

                # cv2.imshow("Depth", depth_display)

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

                    # Run SAM2 on the first frame with a single positive click.
                    # FoundationPose's predict_pose_refine sets the default tensor
                    # type to CUDA and never restores it — reset here so SAM2's
                    # prepare_data does not mix CPU/CUDA tensors.
                    torch.set_default_tensor_type("torch.FloatTensor")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        self._sam_predictor.load_first_frame(rgb_image)
                        input_point = np.array(
                            [[self._clicked_point[0], self._clicked_point[1]]],
                            dtype=np.float32,
                        )
                        input_label = np.array([1], dtype=np.int32)
                        _, _, logits = self._sam_predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=self._sam2_obj_id,
                            points=input_point,
                            labels=input_label,
                            clear_old_points=True,
                        )
                    mask = (
                        (logits[0] > 0.0)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool)
                    )

                    # for i, mask in enumerate(masks):
                    #     cv2.imwrite(f"test_{i}.png", (mask * 255).astype(np.uint8))

                    # visualize the mask from SAM
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

                    # estimate the pose for the first frame
                    self._pose = self._estimator.register(
                        K=self._cam_intrinsic,
                        rgb=rgb_image,
                        depth=depth_image_meter,
                        ob_mask=mask,
                        iteration=self._est_refine_iter,
                    )

                    if self._debug >= 3 and self._mesh is not None:
                        m = self._mesh.copy()
                        m.apply_transform(copy.deepcopy(self._pose))
                        m.export(f"{self._debug_dir}/model_tf.obj")
                        xyz_map = depth2xyzmap(depth_image_meter, self._cam_intrinsic)
                        valid = depth_image_meter >= 0.001
                        pcd = toOpen3dCloud(xyz_map[valid], rgb_image[valid])
                        o3d.io.write_point_cloud(
                            f"{self._debug_dir}/scene_complete.ply", pcd
                        )

                    self._initialized = True

                    # Free SAM2 predictor from memory after first frame
                    del self._sam_predictor
                    self._sam_predictor = None

                    # Reset pose filter for new tracking session
                    if self._use_filter and self._pose_filter is not None:
                        self._pose_filter.reset()

                    # pop out the rgbd data for real time tracking
                    while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                        self.extr_sub_que_dict[self._rgbd_channel].get()

                    # clear the publisher queue
                    if self.percep_pub_que_dict[self._pose_pub_channel]:
                        while not self.percep_pub_que_dict[
                            self._pose_pub_channel
                        ].empty():
                            self.percep_pub_que_dict[self._pose_pub_channel].get()

            else:
                self._pose = self._estimator.track_one(
                    rgb=rgb_image,
                    depth=depth_image_meter,
                    K=self._cam_intrinsic,
                    iteration=self._track_refine_iter,
                )
                self._report_fps()

            if self._pose is not None:
                # Apply SE3 low pass filter if enabled
                if self._use_filter and self._pose_filter is not None:
                    filtered_pose = self._pose_filter.update(self._pose)
                else:
                    filtered_pose = self._pose

                # center_pose takes from object frame in oriented bounding box to camera frame
                center_pose = filtered_pose @ self._to_origin_inv
                self._pose_world = self._cam2world @ center_pose

                # print(self._pose_world)
                # self._pose_world = np.array([[ 0.07414275, -0.03796141,  0.9965249,  -0.36965097],
                #                 [-0.99723276, -0.00827365,  0.07388037, -0.70207579],
                #                 [ 0.00544024, -0.99924498, -0.03846994,  0.022883  ],
                #                 [ 0.          , 0.          , 0.          , 1.        ]])

                if self._fix_x_rotation:
                    x_axis = self._pose_world[:3, 0]
                    y_axis = np.cross(x_axis, [0, 1, 0])
                    y_axis /= np.linalg.norm(y_axis)
                    z_axis = np.cross(x_axis, y_axis)
                    z_axis /= np.linalg.norm(z_axis)
                    self._pose_world[:3, 0] = x_axis
                    self._pose_world[:3, 1] = y_axis
                    self._pose_world[:3, 2] = z_axis

                    center_pose = np.linalg.inv(self._cam2world) @ self._pose_world

                if self._fix_z_rotation:
                    z_axis = self._pose_world[:3, 2]
                    x_axis = np.cross([1, 0, 0], z_axis)
                    x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis /= np.linalg.norm(y_axis)
                    self._pose_world[:3, 0] = x_axis
                    self._pose_world[:3, 1] = y_axis
                    self._pose_world[:3, 2] = z_axis

                    center_pose = np.linalg.inv(self._cam2world) @ self._pose_world

                # publish the pose in named_vec format
                if self._publisher_name == "named_vec":
                    pose_out = np.concatenate(
                        [self._pose_world[:3, 3], self._pose_world[:3, :3].flatten()],
                        axis=0,
                    ).reshape(1, 12)
                    out_data = NamedVecListData(
                        num_vecs=1, vec_dim=12, name=self._object_name
                    )
                    out_data.set_data(
                        rgbd_data.get_time(), [self._object_name], pose_out
                    )

                    if self.percep_pub_que_dict[self._pose_pub_channel]:
                        self.percep_pub_que_dict[self._pose_pub_channel].put(out_data)

                if self._visualize:
                    # draw the pose
                    rgb_vis = rgb_image.copy()
                    draw_posed_3d_box(
                        self._cam_intrinsic,
                        rgb_vis,
                        center_pose,
                        self._bbox,
                        line_color=(0, 255, 0),
                    )
                    rgb_vis = draw_xyz_axis(
                        rgb_vis,
                        ob_in_cam=center_pose,
                        K=self._cam_intrinsic,
                        thickness=3,
                        transparency=0,
                        is_input_rgb=True,
                    )
                    self._vis_image(rgb_vis, self._window_name)
            # else:
            #     logging.warning("[FoundationPosePerception] Color image is None.")
        # raise NotImplementedError("Subclasses should implement this method.")

    def _report_fps(self):
        """Count one processed tracking frame; every fps_report_every frames,
        print the aggregate FPS over that window."""
        if self._fps_report_every <= 0:
            return
        import time
        if self._fps_window_start is None:
            self._fps_window_start = time.time()
        self._fps_frame_count += 1
        if self._fps_frame_count >= self._fps_report_every:
            dt = time.time() - self._fps_window_start
            fps = self._fps_frame_count / dt if dt > 0 else float("inf")
            print(f"[FoundationPoseTracking] {fps:.1f} FPS")
            self._fps_frame_count = 0
            self._fps_window_start = time.time()

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

    def _vis_image(self, image, window_name="robot_view"):
        # Convert RGB to BGR for OpenCV
        if image.shape[-1] == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image
        cv2.imshow(window_name, bgr_image)
        cv2.waitKey(1)
