"""FoundationPose tracking with SAM2 streaming mask refinement.

Unlike FoundationPoseTracking (which only uses SAM2 for the first frame),
this module runs SAM2CameraPredictor in streaming mode and applies the
per-frame mask to zero out background in RGB/depth before feeding
FoundationPose's track_one. This stabilizes tracking in cluttered scenes
and enables auto re-registration when observations drift from the mesh.

Requires the segment-anything-2-real-time fork (SAM2CameraPredictor).
"""

import copy
import logging
import os
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
import nvdiffrast.torch as dr

from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_camera_predictor

from perception.BasePerception import BasePerception
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.NamedVecListData import NamedVecListData
from utils.visualization.bounding_box import draw_posed_3d_box, draw_xyz_axis
from utils.lie.se3 import invSE3
from utils.filtering.SE3LowPassFilter import SE3LowPassFilter

from third_party.FoundationPose.Utils import depth2xyzmap, toOpen3dCloud
from third_party.FoundationPose.estimater import FoundationPose
from third_party.FoundationPose.learning.training.predict_score import ScorePredictor
from third_party.FoundationPose.learning.training.predict_pose_refine import (
    PoseRefinePredictor,
)


class FoundationPoseTrackingSAM2(BasePerception):
    """FoundationPose tracking with continuous SAM2 mask refinement."""

    def __init__(self, config):
        super().__init__(config)

        self._mesh_path = config.get("mesh_file", None)
        self._mesh = trimesh.load(self._mesh_path, force="mesh")
        # FoundationPose crop/refine path is float32; trimesh defaults to float64.
        self._mesh.vertices = np.asarray(self._mesh.vertices, dtype=np.float32)
        self._to_origin, extents = trimesh.bounds.oriented_bounds(self._mesh)
        self._to_origin_inv = np.linalg.inv(self._to_origin)
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        self._pose: Optional[np.ndarray] = None
        self._pose_world: Optional[np.ndarray] = None

        self._use_filter = config.get("pose_filter").get("use_filter", True)
        if self._use_filter:
            filter_alpha = config.get("pose_filter").get("alpha", 0.1)
            filter_window_size = config.get("pose_filter").get("window_size", 0)
            self._pose_filter: Optional[SE3LowPassFilter] = SE3LowPassFilter(
                alpha=filter_alpha, window_size=filter_window_size
            )
        else:
            self._pose_filter = None
        self._fix_x_rotation = config.get("pose_filter").get("fix_x_rotation", False)
        self._fix_z_rotation = config.get("pose_filter").get("fix_z_rotation", False)

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
            for _n in (
                "FoundationPose",
                "sam2",
                "SAM2",
                "predict_score",
                "predict_pose_refine",
            ):
                logging.getLogger(_n).setLevel(logging.WARNING)
        self._fps_frame_count = 0
        self._fps_window_start = None
        self._last_score = None

        # Fixed input shapes — let cuDNN cache the best kernel per shape.
        torch.backends.cudnn.benchmark = True

        # Auto-reset. Triggered by FoundationPose's tracking score (computed
        # here in _track_score, not inside track_one, to keep third_party
        # untouched): on outliers (mask drift, occlusion) the score blows up
        # past score_threshold. After streak consecutive lost frames we
        # re-register from the current SAM2 mask.
        auto_cfg = config.get("auto_reset", {})
        self._auto_reset = auto_cfg.get("enable", False)
        self._auto_reset_score_threshold = auto_cfg.get("score_threshold", 100.0)
        self._auto_reset_streak = auto_cfg.get("streak", 10)
        self._trigger_streak = 0

        self._glctx = dr.RasterizeCudaContext()
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        # Stock FoundationPose.__init__ does not take min_n_views/inplane_step;
        # those are applied via make_rotation_grid after construction.
        self._estimator = FoundationPose(
            model_pts=self._mesh.vertices,
            model_normals=self._mesh.vertex_normals,
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

        # SAM2 streaming tracker
        sam2_cfg = config.get("sam2", {})
        self._sam2_config_file = sam2_cfg.get(
            "config_file", "configs/sam2.1/sam2.1_hiera_t.yaml"
        )
        self._sam2_hf_repo = sam2_cfg.get("hf_repo", "facebook/sam2.1-hiera-tiny")
        self._sam2_hf_filename = sam2_cfg.get("hf_filename", "sam2.1_hiera_tiny.pt")
        self._sam2_mask_rgb = sam2_cfg.get("mask_rgb", True)
        self._sam2_mask_depth = sam2_cfg.get("mask_depth", True)
        self._sam2_obj_id = sam2_cfg.get("obj_id", 1)
        self._sam2_predictor = None

        # Scene ROI: "x,y,w,h" to crop every RGBD frame before models see it.
        # If null, user draws it interactively on first frame.
        self._scene_roi = self._parse_scene_roi(config.get("scene_roi", None))
        self._K_cropped: Optional[np.ndarray] = None

        # Init state machine: need_roi -> wait_click -> preview -> (accept) tracking
        self._init_phase = "need_roi" if self._scene_roi is None else "wait_click"
        self._preview_rgb: Optional[np.ndarray] = None
        self._preview_depth: Optional[np.ndarray] = None
        self._preview_mask: Optional[np.ndarray] = None
        self._preview_point: Optional[tuple] = None
        self._last_key = -1

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
        self._visualize_sam = config.get("visualize_sam", False)
        self._clicked_point = None
        self._window_name = "FoundationPoseTrackingSAM2"

    def initialize(self):
        self._init_gui()

        logging.info(f"initializing SAM2CameraPredictor ({self._sam2_config_file})")
        ckpt_path = hf_hub_download(
            repo_id=self._sam2_hf_repo, filename=self._sam2_hf_filename
        )
        self._sam2_predictor = build_sam2_camera_predictor(
            self._sam2_config_file, ckpt_path, device="cuda", vos_optimized=True
        )
        logging.info("SAM2CameraPredictor initialized")

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
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    @staticmethod
    def _parse_scene_roi(s):
        if s is None:
            return None
        parts = [int(v) for v in str(s).split(",")]
        if len(parts) != 4:
            raise ValueError(f"scene_roi must be 'x,y,w,h', got {s!r}")
        return tuple(parts)

    def _crop_rgbd(self, rgb, depth):
        sx, sy, sw, sh = self._scene_roi
        return rgb[sy : sy + sh, sx : sx + sw], depth[sy : sy + sh, sx : sx + sw]

    @staticmethod
    def _crop_K(K, sx, sy):
        Kc = K.copy()
        Kc[0, 2] -= sx
        Kc[1, 2] -= sy
        return Kc

    def _select_roi_interactive(self, rgb):
        win = "Draw SCENE ROI (ENTER to confirm, C to cancel)"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        roi = cv2.selectROI(win, bgr, showCrosshair=False, fromCenter=False)
        cv2.destroyWindow(win)
        x, y, w, h = [int(v) for v in roi]
        if w == 0 or h == 0:
            raise RuntimeError("Empty scene ROI")
        return (x, y, w, h)

    def _resize_mask_if_needed(self, mask, target_hw):
        h, w = target_hw
        if mask.shape == (h, w):
            return mask
        return cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    def _run_sam_prompt(self, rgb, point, reload_first_frame):
        """Run SAM2 on `rgb` with a single positive click.
        @reload_first_frame: call load_first_frame first (use when rgb differs
            from the previously loaded first frame).
        """
        torch.set_default_tensor_type("torch.FloatTensor")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if reload_first_frame:
                self._sam2_predictor.load_first_frame(rgb)
            pts = np.array([[point[0], point[1]]], dtype=np.float32)
            lbl = np.array([1], dtype=np.int32)
            _, _, logits = self._sam2_predictor.add_new_prompt(
                frame_idx=0,
                obj_id=self._sam2_obj_id,
                points=pts,
                labels=lbl,
                clear_old_points=True,
            )
        mask = (logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(bool)
        return self._resize_mask_if_needed(mask, rgb.shape[:2])

    def _make_preview_vis(self, rgb, mask, point):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        overlay[mask] = (0, 255, 0)
        vis = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
        cv2.circle(vis, point, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            vis,
            "click: re-mask   r: reset   y/Enter: accept",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    def _sam2_step(self, rgb_image):
        """Run the streaming predictor on one frame; returns a bool (H,W) mask."""
        # FoundationPose's predict_pose_refine sets torch.set_default_tensor_type
        # to CUDA and never restores it — this leaks into SAM2's perpare_data
        # where it creates img on CPU but img_mean/img_std on CUDA. Reset here;
        # FoundationPose re-sets it on its next predict() call.
        torch.set_default_tensor_type("torch.FloatTensor")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, mask_logits = self._sam2_predictor.track(rgb_image)
        # mask_logits: list/stack over objects; take the first (we track one).
        mask = (mask_logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(bool)
        return mask

    def _apply_mask(self, rgb, depth, mask):
        rgb_out = rgb
        depth_out = depth
        if self._sam2_mask_rgb:
            rgb_out = rgb.copy()
            rgb_out[~mask] = 0
        if self._sam2_mask_depth:
            depth_out = depth.copy()
            depth_out[~mask] = 0
        return rgb_out, depth_out

    def _track_score(self, rgb, depth):
        """FoundationPose scorer value for the pose track_one just produced.

        Kept out of third_party/FoundationPose.track_one (which is gitignored
        and must stay at the NVlabs original) — we re-score here using the raw
        centered-mesh pose track_one cached in estimator.pose_last, matching
        what an in-estimator scorer call would use. Score normally peaks for a
        good fit but blows up on outliers (mask drift, occlusion); a value above
        the threshold means tracking is lost. Returns a float, or None if it
        cannot be computed.
        """
        est = self._estimator
        pose_last = getattr(est, "pose_last", None)
        if pose_last is None:
            return None
        ob_in_cams = pose_last.reshape(1, 4, 4).data.cpu().numpy()
        scores, _ = est.scorer.predict(
            mesh=est.mesh,
            rgb=rgb,
            depth=depth,
            K=self._K_cropped,
            ob_in_cams=ob_in_cams,
            normal_map=None,
            mesh_tensors=est.mesh_tensors,
            glctx=est.glctx,
            mesh_diameter=est.diameter,
            get_vis=False,
        )
        return float(scores[0])

    def _process(self):
        if self.extr_sub_que_dict[self._rgbd_channel].empty():
            return

        while not self.extr_sub_que_dict[self._rgbd_channel].empty():
            rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

        rgb_full = rgbd_data.get_rgb_image()
        depth_full_meter = (
            rgbd_data.get_depth_image().astype(np.float32) / self._depth_factor
        )

        # ----- Init: draw ROI, then click + preview + accept/reset loop -----
        if not self._initialized:
            if self._init_phase == "need_roi":
                # Blocking selectROI on a live frame. Sets _scene_roi and K_cropped.
                self._scene_roi = self._select_roi_interactive(rgb_full)
                sx, sy, _, _ = self._scene_roi
                self._K_cropped = self._crop_K(self._cam_intrinsic, sx, sy)
                self._init_phase = "wait_click"
                return

            if self._K_cropped is None:
                sx, sy, _, _ = self._scene_roi
                self._K_cropped = self._crop_K(self._cam_intrinsic, sx, sy)

            rgb_crop, depth_crop = self._crop_rgbd(rgb_full, depth_full_meter)

            if self._init_phase == "wait_click":
                self._vis_image(rgb_crop.copy(), self._window_name)
                if self._clicked_point is None:
                    return
                pt = self._clicked_point
                self._clicked_point = None

                self._preview_rgb = rgb_crop.copy()
                self._preview_depth = depth_crop.copy()
                self._preview_point = pt
                self._preview_mask = self._run_sam_prompt(
                    self._preview_rgb,
                    pt,
                    reload_first_frame=True,
                )
                self._init_phase = "preview"
                return

            if self._init_phase == "preview":
                vis = self._make_preview_vis(
                    self._preview_rgb,
                    self._preview_mask,
                    self._preview_point,
                )
                self._vis_image(vis, self._window_name)

                # New click while previewing: re-run SAM on the frozen frame.
                if self._clicked_point is not None:
                    pt = self._clicked_point
                    self._clicked_point = None
                    self._preview_point = pt
                    self._preview_mask = self._run_sam_prompt(
                        self._preview_rgb,
                        pt,
                        reload_first_frame=False,
                    )
                    return

                key = self._last_key
                if key == ord("r"):
                    self._init_phase = "wait_click"
                    self._preview_rgb = None
                    self._preview_depth = None
                    self._preview_mask = None
                    self._preview_point = None
                    self._last_key = -1
                    return
                if key in (ord("y"), 13, 10, 32):  # y / Enter / LF / Space
                    self._last_key = -1
                    self._pose = self._estimator.register(
                        K=self._K_cropped,
                        rgb=self._preview_rgb,
                        depth=self._preview_depth,
                        ob_mask=self._preview_mask,
                        iteration=self._est_refine_iter,
                    )

                    if self._debug >= 3:
                        m = self._mesh.copy()
                        m.apply_transform(copy.deepcopy(self._pose))
                        m.export(f"{self._debug_dir}/model_tf.obj")
                        xyz_map = depth2xyzmap(self._preview_depth, self._K_cropped)
                        valid = self._preview_depth >= 0.001
                        pcd = toOpen3dCloud(xyz_map[valid], self._preview_rgb[valid])
                        o3d.io.write_point_cloud(
                            f"{self._debug_dir}/scene_complete.ply", pcd
                        )

                    self._initialized = True
                    if self._use_filter and self._pose_filter is not None:
                        self._pose_filter.reset()

                    while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                        self.extr_sub_que_dict[self._rgbd_channel].get()
                    if self.percep_pub_que_dict[self._pose_pub_channel]:
                        while not self.percep_pub_que_dict[
                            self._pose_pub_channel
                        ].empty():
                            self.percep_pub_que_dict[self._pose_pub_channel].get()
                return

        # ----- Tracking: crop every frame, run SAM + FoundationPose -----
        rgb_crop, depth_crop = self._crop_rgbd(rgb_full, depth_full_meter)
        mask = self._sam2_step(rgb_crop)
        mask = self._resize_mask_if_needed(mask, rgb_crop.shape[:2])

        rgb_in, depth_in = self._apply_mask(rgb_crop, depth_crop, mask)
        self._pose = self._estimator.track_one(
            rgb=rgb_in,
            depth=depth_in,
            # rgb=rgb_crop,
            # depth=depth_crop,
            K=self._K_cropped,
            iteration=self._track_refine_iter,
        )
        self._report_fps()

        # Score-based lost detection (see _track_score). On outliers the score
        # blows up past score_threshold; treat that as lost and re-register
        # after streak consecutive lost frames.
        if self._auto_reset and self._pose is not None:
            score = self._track_score(rgb_in, depth_in)
            if score is not None:
                self._last_score = score
                if score > self._auto_reset_score_threshold:
                    self._trigger_streak += 1
                else:
                    self._trigger_streak = 0
                if self._trigger_streak >= self._auto_reset_streak:
                    logging.warning(
                        f"auto-reset: tracking lost (score {score:.1f} > "
                        f"{self._auto_reset_score_threshold}) for "
                        f"{self._trigger_streak} frames -> re-register"
                    )
                    self._pose = self._estimator.register(
                        K=self._K_cropped,
                        rgb=rgb_crop,
                        depth=depth_crop,
                        ob_mask=mask,
                        iteration=self._est_refine_iter,
                    )
                    self._trigger_streak = 0

        if self._pose is None:
            return

        if self._use_filter and self._pose_filter is not None:
            filtered_pose = self._pose_filter.update(self._pose)
        else:
            filtered_pose = self._pose

        center_pose = filtered_pose @ self._to_origin_inv
        self._pose_world = self._cam2world @ center_pose

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

        if self._publisher_name == "named_vec":
            pose_out = np.concatenate(
                [self._pose_world[:3, 3], self._pose_world[:3, :3].flatten()],
                axis=0,
            ).reshape(1, 12)
            out_data = NamedVecListData(num_vecs=1, vec_dim=12, name=self._object_name)
            out_data.set_data(rgbd_data.get_time(), [self._object_name], pose_out)
            if self.percep_pub_que_dict[self._pose_pub_channel]:
                self.percep_pub_que_dict[self._pose_pub_channel].put(out_data)

        if self._visualize:
            rgb_vis = rgb_crop.copy()
            draw_posed_3d_box(
                self._K_cropped,
                rgb_vis,
                center_pose,
                self._bbox,
                line_color=(0, 255, 0),
            )
            rgb_vis = draw_xyz_axis(
                rgb_vis,
                ob_in_cam=center_pose,
                K=self._K_cropped,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            if self._initialized and self._visualize_sam:
                # Draw mask contour on top
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                rgb_vis_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
                rgb_vis_bgr = cv2.drawContours(
                    rgb_vis_bgr, contours, -1, (0, 255, 255), 2
                )
                rgb_vis = cv2.cvtColor(rgb_vis_bgr, cv2.COLOR_BGR2RGB)
            self._vis_image(rgb_vis, self._window_name)

    def _report_fps(self):
        """Count one processed tracking frame; every fps_report_every frames,
        print the aggregate FPS (and latest tracking score) over that window."""
        if self._fps_report_every <= 0:
            return
        import time

        if self._fps_window_start is None:
            self._fps_window_start = time.time()
        self._fps_frame_count += 1
        if self._fps_frame_count >= self._fps_report_every:
            dt = time.time() - self._fps_window_start
            fps = self._fps_frame_count / dt if dt > 0 else float("inf")
            score_str = (
                "" if self._last_score is None else f" | score={self._last_score:.1f}"
            )
            print(f"[FoundationPoseTrackingSAM2] {fps:.1f} FPS{score_str}")
            self._fps_frame_count = 0
            self._fps_window_start = time.time()

    def _init_gui(self):
        self._clicked_point = None
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked_point = (x, y)
            print(f"Clicked pixel: ({x}, {y})")

    def _vis_image(self, image, window_name="robot_view"):
        if image.shape[-1] == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image
        cv2.imshow(window_name, bgr_image)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self._last_key = key
