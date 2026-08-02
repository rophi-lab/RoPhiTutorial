"""
Multi-view FoundationPose pose tracking via LCM.

Extends the single-view SAM2 variant to N camera views to be robust against
occlusion (e.g. the hand covering the object in one view). scorer / refiner /
glctx / mesh are shared across views (they are stateless w.r.t. the tracked
pose); each view keeps its own FoundationPose estimator (its own pose_last, in
that view's camera frame), its own SAM2 streaming predictor, and its own camera
parameters.

Every frame, each view runs SAM2 + track_one and is re-scored. The most
trustworthy view (guarded by an absolute lost-threshold and a cross-view margin)
is selected as the winner; its pose is lifted to world and pushed back down into
the losing views' pose_last so they re-sync instead of drifting. The winner's
world pose is filtered and published.

NOTE on scoring: FoundationPose's scorer output is a learned per-image ranking
logit, not a calibrated absolute metric, so cross-view comparison is not
guaranteed clean. score_direction / lost_threshold / resync_margin are all
config-exposed; log per-view scores on first runs to fix them empirically.
"""

from typing import List, Optional
from contextlib import contextmanager
import copy
import logging
import os
import time

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
import open3d as o3d

from third_party.FoundationPose.Utils import depth2xyzmap, toOpen3dCloud
from third_party.FoundationPose.gsplat_renderer import make_gaussian_tensors
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

# FoundationPose's Utils.py calls logging.basicConfig(level=INFO) on the root
# logger at import time, so its per-frame logging.info() spam floods the console.
# Silencing is applied in __init__ (config-driven) via _apply_quiet_logs().

try:
    import cv2  # OpenCV is required for visualization
except ImportError as e:
    raise ImportError(
        "OpenCV (cv2) is required for FoundationPoseTrackingMultiView visualization. "
        "Please install it with 'pip install opencv-python'."
    ) from e


def _to_origin_from_neighbor_mesh(splat_path, means):
    """Return (to_origin, extents) canonicalizing the object frame from the mesh
    sitting next to a gaussian-splat file, falling back to the gaussians' own
    oriented bounds when no mesh is found.

    The published object pose ("bb" frame) must match what pose consumers
    (controller bb->cad, MeshGraspVisualizer) reconstruct via
    trimesh.bounds.oriented_bounds(mesh). Because oriented_bounds() picks
    principal-axis signs arbitrarily, computing it on the gaussian means instead
    can yield a frame flipped 180 deg from the mesh's, flipping the rendered
    mesh. Deriving to_origin from the mesh guarantees agreement (valid because
    the PCA-aligned splats share the mesh's native frame).
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


class _ViewState:
    """Per-view mutable state. One instance per camera channel."""

    def __init__(self, channel: str):
        self.channel = channel
        self.window_name = f"FoundationPoseMultiView [{channel}]"

        # camera parameters (filled during initialize())
        self.cam_intrinsic: Optional[np.ndarray] = None
        self.cam2world: Optional[np.ndarray] = None
        self.world2cam: Optional[np.ndarray] = None
        self.depth_factor: Optional[float] = None

        # per-view models
        self.estimator: Optional[FoundationPose] = None
        self.sam2_predictor = None

        # scene ROI / cropped intrinsics
        self.scene_roi: Optional[tuple] = None
        self.K_cropped: Optional[np.ndarray] = None

        # register / preview state machine
        self.init_phase = "need_roi"  # need_roi -> wait_click -> preview -> registered
        self.registered = False
        self.clicked_point: Optional[tuple] = None
        self.last_key = -1
        self.preview_rgb: Optional[np.ndarray] = None
        self.preview_depth: Optional[np.ndarray] = None
        self.preview_mask: Optional[np.ndarray] = None
        self.preview_point: Optional[tuple] = None

        # latest per-frame track results (for winner selection / publish)
        self.pose: Optional[np.ndarray] = None  # ob_in_cam (oriented-bbox origin)
        self.score: Optional[float] = None
        self.last_rgbd_time = None


class FoundationPoseTrackingMultiView(BasePerception):
    """Multi-view FoundationPose pose tracking perception module."""

    def __init__(self, config):
        super().__init__(config)

        # ---- shared object / mesh, or splat_file for a Gaussian-Splat object
        # model (mutually exclusive with mesh_file) ----
        self._splat_path = config.get("splat_file", None)
        self._is_gaussian = bool(self._splat_path)

        if self._is_gaussian:
            self._mesh_path = None
            self._mesh = None
            self._gaussian_tensors = make_gaussian_tensors(self._splat_path, device="cuda")
            means = self._gaussian_tensors["means"].detach().cpu().numpy()
            # Canonicalize the published (bb) frame from the object MESH's
            # oriented bounds, NOT the gaussians'. trimesh.oriented_bounds()
            # picks principal-axis SIGNS arbitrarily, so oriented_bounds(means)
            # and oriented_bounds(mesh) can differ by a 180-deg flip even though
            # the PCA-aligned splats share the mesh's native frame. Every
            # consumer (controller bb->cad, MeshGraspVisualizer) applies
            # oriented_bounds(mesh) as cad_to_bb, so publishing the gaussians'
            # own OBB frame flips the mesh. Deriving to_origin from the mesh here
            # makes the published bb frame match what consumers expect. Falls
            # back to the gaussians' OBB if no mesh sits beside the splat file.
            self._to_origin, extents = _to_origin_from_neighbor_mesh(
                self._splat_path, means
            )
            self._model_pts, self._model_normals = means, None
        else:
            self._mesh_path = config.get("mesh_file", None)
            self._mesh = trimesh.load(self._mesh_path)
            self._gaussian_tensors = None
            self._to_origin, extents = trimesh.bounds.oriented_bounds(self._mesh)
            self._model_pts, self._model_normals = self._mesh.vertices, self._mesh.vertex_normals

        self._to_origin_inv = np.linalg.inv(self._to_origin)
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        self._pose_world: Optional[np.ndarray] = None

        # ---- shared pose filter (applied once, on the published world pose) ----
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

        # ---- FoundationPose params ----
        self._est_refine_iter = config.get("est_refine_iter", 5)
        self._track_refine_iter = config.get("track_refine_iter", 2)
        # Register pose-hypothesis grid size. Defaults reproduce FoundationPose's
        # 40 views x 6 in-plane rots (~252 before symmetry clustering). Lower
        # register_min_n_views / raise register_inplane_step to speed up register.
        self._register_min_n_views = config.get("register_min_n_views", 40)
        self._register_inplane_step = config.get("register_inplane_step", 60)
        self._debug = config.get("debug", 0)
        self._debug_dir = config.get("debug_dir", "./debug")

        # ---- logging / FPS ----
        # quiet_logs: silence FoundationPose/SAM2 per-frame INFO spam (they call
        # logging.basicConfig on the root logger at import). fps_report_every:
        # print aggregate tracking FPS every N processed frames (0 = never).
        self._quiet_logs = config.get("quiet_logs", True)
        self._fps_report_every = config.get("fps_report_every", 30)
        self._apply_quiet_logs()
        self._fps_frame_count = 0
        self._fps_window_start = None

        # profile_stages: break the per-frame FPS line down into SAM2 vs
        # FoundationPose (track_one) vs scorer wall-clock, summed over all views.
        # Uses torch.cuda.synchronize() so async GPU work is timed correctly
        # (adds a small sync overhead -- turn off once the bottleneck is known).
        self._profile_stages = config.get("profile_stages", False)
        self._prof = {"sam2": 0.0, "track": 0.0, "score": 0.0}

        torch.backends.cudnn.benchmark = True

        # ---- shared models: scorer / refiner / glctx ----
        self._glctx = dr.RasterizeCudaContext()
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()

        # Render/crop padding around the object. The crop window radius is
        # mesh_diameter * crop_ratio / 2, so a larger value leaves more margin
        # around the object in the rendered+observed crop (both the network
        # inputs and the debug render overlay). The released weights train at
        # 1.2; read fresh at predict time, so overriding both nets' cfg here
        # takes effect. NOTE: pushing far from ~1.2 shrinks the object in the
        # fixed input_resize window and can reduce pose accuracy -- keep modest
        # (≈1.4-1.6). Omit / null to leave the model default untouched.
        crop_ratio = config.get("crop_ratio", None)
        if crop_ratio is not None:
            self._scorer.cfg["crop_ratio"] = float(crop_ratio)
            self._refiner.cfg["crop_ratio"] = float(crop_ratio)
            logging.warning(
                "[FoundationPose] crop_ratio overridden to %.3f (default 1.2); "
                "larger = more crop padding but the object shrinks in the fixed "
                "input window, which can degrade pose accuracy.",
                float(crop_ratio),
            )

        # ---- SAM2 config (one predictor is built per view in initialize) ----
        sam2_cfg = config.get("sam2", {})
        self._sam2_config_file = sam2_cfg.get(
            "config_file", "configs/sam2.1/sam2.1_hiera_t.yaml"
        )
        self._sam2_hf_repo = sam2_cfg.get("hf_repo", "facebook/sam2.1-hiera-tiny")
        self._sam2_hf_filename = sam2_cfg.get("hf_filename", "sam2.1_hiera_tiny.pt")
        self._sam2_checkpoint = sam2_cfg.get("checkpoint", None)
        self._sam2_mask_rgb = sam2_cfg.get("mask_rgb", True)
        self._sam2_mask_depth = sam2_cfg.get("mask_depth", True)
        self._sam2_obj_id = sam2_cfg.get("obj_id", 1)

        # torch.compile toggles (see initialize()). The image encoder is the
        # per-frame bottleneck, so it is the one to enable first. Others target
        # the lighter decoder/memory stages. First forward is slow (autotune).
        compile_cfg = sam2_cfg.get("compile", {})
        self._sam2_compile_image_encoder = compile_cfg.get("image_encoder", False)
        self._sam2_compile_memory_attention = compile_cfg.get("memory_attention", False)
        self._sam2_compile_memory_encoder = compile_cfg.get("memory_encoder", False)
        self._sam2_compile_mask_decoder = compile_cfg.get("mask_decoder", False)
        self._sam2_compile_prompt_encoder = compile_cfg.get("prompt_encoder", False)


        # ---- multi-view fusion params ----
        mv_cfg = config.get("multiview", {})
        # Winner selection direction: "max" -> higher score = better fit
        # (register()'s argsort(descending=True)); "min" -> lower is better.
        self._score_direction = mv_cfg.get("score_direction", "max")
        # Lost detection is SEPARATE from winner direction: even when higher =
        # better, an outlier blows the re-scored pose up past the threshold. So
        # lost_side defaults to "above" (score > threshold => lost). Views judged
        # lost are excluded from winner candidacy.
        self._lost_threshold = mv_cfg.get("lost_threshold", None)
        self._lost_side = mv_cfg.get("lost_side", "above")
        # Only re-sync losing views when the winner beats the runner-up by at
        # least this margin (guards against uncalibrated cross-view score noise).
        self._resync_margin = mv_cfg.get("resync_margin", 0.0)
        self._resync_enable = mv_cfg.get("resync_enable", True)

        # ---- auto re-register fallback ----
        # When every view is lost (no trustworthy winner) for `streak`
        # consecutive frames -- e.g. the object moved too fast and all trackers
        # drifted -- re-register each view from its current SAM2 mask.
        auto_cfg = config.get("auto_reset", {})
        self._auto_reset = auto_cfg.get("enable", True)
        self._auto_reset_streak = auto_cfg.get("streak", 3)
        self._all_lost_streak = 0

        # per-view scene ROI list: "x,y,w,h" strings, one per channel (or null).
        self._scene_roi_cfg = config.get("scene_roi", None)

        # ---- channels ----
        sub_cfg = config.get("sub_manager", {})
        channels = sub_cfg.get("rgbd_channels", None)
        if channels is None:
            # tolerate single-view style config
            channels = [sub_cfg.get("rgbd_channel", "d455_1")]
        self._channels: List[str] = list(channels)
        self._views: List[_ViewState] = [_ViewState(ch) for ch in self._channels]

        # ---- publisher ----
        self._object_name = config.get("object_name", "predefined_obj")
        self._publisher_name = config["pub_manager"]["name"]
        if self._publisher_name == "named_vec":
            self._pose_pub_channel = config["pub_manager"]["named_vec_list_channel"]

        self._visualize = config.get("visualize", False)
        self._visualize_sam = config.get("visualize_sam", False)
        # Live per-frame gsplat/mesh render overlay: show the refiner's rendered
        # model (column A) beside the observed crop (column B) every tracking
        # frame, in a separate "[render]" window per view. Costs an extra render
        # per frame (the refiner re-renders to build the vis), so keep it off in
        # production. See _track_view_maskless (bumps the estimator debug level
        # just for the track_one call so it emits extra['vis']).
        self._visualize_render = config.get("visualize_render", False)

        self._initialized = False  # True once all views registered

    # --------------------------------------------------------------- logging

    def _apply_quiet_logs(self):
        """Silence FoundationPose / SAM2 per-frame INFO spam. Both call
        logging.basicConfig on the root logger at import time; raise the root
        level to WARNING so only warnings/errors get through while we print our
        own FPS line via print()."""
        if not self._quiet_logs:
            return
        logging.getLogger().setLevel(logging.WARNING)
        for name in ("FoundationPose", "sam2", "SAM2", "predict_score",
                     "predict_pose_refine"):
            logging.getLogger(name).setLevel(logging.WARNING)

    @contextmanager
    def _stage_timer(self, key):
        """Accumulate GPU wall-clock for a pipeline stage into self._prof[key].
        No-op (near-zero overhead) when profiling is off. Synchronizes CUDA so
        async kernels are attributed to the right stage."""
        if not self._profile_stages:
            yield
            return
        import time
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self._prof[key] += time.time() - t0

    def _report_fps(self):
        """Count one processed tracking frame; every fps_report_every frames,
        print the aggregate FPS over that window (and, if profiling is on, the
        per-stage SAM2 / track_one / scorer time share summed over all views)."""
        if self._fps_report_every <= 0:
            return
        import time
        if self._fps_window_start is None:
            self._fps_window_start = time.time()
        self._fps_frame_count += 1
        if self._fps_frame_count >= self._fps_report_every:
            dt = time.time() - self._fps_window_start
            fps = self._fps_frame_count / dt if dt > 0 else float("inf")
            # Include per-view scores so the lost_threshold can be tuned without
            # the GUI (e.g. "upper_camera=42.1 side_camera=151.7*" -- * = lost).
            score_str = " ".join(
                f"{v.channel}="
                + ("-" if v.score is None else f"{v.score:.1f}")
                + ("*" if (v.score is not None and self._is_lost(v.score)) else "")
                for v in self._views
            )
            line = f"[FoundationPoseMultiView] {fps:.1f} FPS | {score_str}"
            if self._profile_stages:
                n = self._fps_frame_count
                # per-frame ms for each stage (summed over all views per frame)
                sam2 = 1e3 * self._prof["sam2"] / n
                trk = 1e3 * self._prof["track"] / n
                scr = 1e3 * self._prof["score"] / n
                line += (
                    f" | SAM2 {sam2:.0f}ms  track {trk:.0f}ms  score {scr:.0f}ms"
                    f"  (FP {trk + scr:.0f}ms)"
                )
                self._prof = {"sam2": 0.0, "track": 0.0, "score": 0.0}
            print(line)
            self._fps_frame_count = 0
            self._fps_window_start = time.time()

    # ------------------------------------------------------------------ init

    def initialize(self):
        self._init_gui()

        # Build one SAM2 predictor per view and one FoundationPose estimator per
        # view (sharing scorer / refiner / glctx). Wait for each view's camera
        # parameters.
        if self._sam2_checkpoint:
            ckpt_path = self._sam2_checkpoint
        else:
            ckpt_path = hf_hub_download(
                repo_id=self._sam2_hf_repo, filename=self._sam2_hf_filename
            )
        logging.info(f"SAM2 checkpoint: {ckpt_path}")

        # torch.compile overrides. The image encoder (Hiera backbone) is the
        # dominant per-frame cost; compiling it targets the bottleneck directly
        # without the resolution loss of view concatenation. The VOS predictor
        # additionally exposes memory/attention/decoder compile flags. All are
        # passed as Hydra model overrides -> no edits to the SAM2 fork. First
        # forward per input shape is slow (autotune warmup); steady state is
        # faster.
        compile_overrides = []
        if self._sam2_compile_image_encoder:
            compile_overrides.append("++model.compile_image_encoder=true")
        if self._sam2_compile_memory_attention:
            compile_overrides.append("++model.compile_memory_attention=true")
        if self._sam2_compile_memory_encoder:
            compile_overrides.append("++model.compile_memory_encoder=true")
        if self._sam2_compile_mask_decoder:
            compile_overrides.append("++model.compile_mask_decoder=true")
        if self._sam2_compile_prompt_encoder:
            compile_overrides.append("++model.compile_prompt_encoder=true")
        if compile_overrides:
            logging.info(f"SAM2 torch.compile overrides: {compile_overrides}")

        for v in self._views:
            v.sam2_predictor = build_sam2_camera_predictor(
                self._sam2_config_file, ckpt_path, device="cuda",
                vos_optimized=True, hydra_overrides_extra=compile_overrides,
            )
            v.estimator = FoundationPose(
                model_pts=self._model_pts,
                model_normals=self._model_normals,
                mesh=self._mesh,
                scorer=self._scorer,
                refiner=self._refiner,
                debug_dir=self._debug_dir,
                debug=self._debug,
                glctx=self._glctx,
            )
            v.estimator.make_rotation_grid(
                min_n_views=self._register_min_n_views,
                inplane_step=self._register_inplane_step,
            )
            v.scene_roi = self._parse_scene_roi_for(v.channel)
            v.init_phase = "need_roi" if v.scene_roi is None else "wait_click"
            logging.info(f"[{v.channel}] SAM2 + FoundationPose initialized")

        # wait for all cameras' intrinsics/extrinsics
        pending = {v.channel: v for v in self._views}
        while pending:
            for ch, v in list(pending.items()):
                info_q = self.extr_sub_que_dict[ch + "_info"]
                if not info_q.empty():
                    cam_info = info_q.get()
                    if isinstance(cam_info, CameraInfoData):
                        # float32: CameraInfoData builds K as float64, but FoundationPose
                        # runs in cuda.FloatTensor. A float64 K makes register() produce
                        # float64 ob_in_cams, which then clashes with the float32 K in
                        # compute_crop_window_tf_batch (K@pts: float != double).
                        v.cam_intrinsic = cam_info.get_intrinsic().astype(np.float32)
                        v.cam2world = invSE3(cam_info.get_extrinsic())
                        v.world2cam = np.linalg.inv(v.cam2world)
                        v.depth_factor = cam_info.get_depth_factor()
                        print(f"[{ch}] camera parameters initialized.")
                        del pending[ch]

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    def _init_gui(self):
        for v in self._views:
            cv2.namedWindow(v.window_name)
            cv2.setMouseCallback(v.window_name, self._make_mouse_callback(v))

    def _make_mouse_callback(self, view: _ViewState):
        def _cb(event, x, y, flags, param=None):
            if event == cv2.EVENT_LBUTTONDOWN:
                view.clicked_point = (x, y)
                print(f"[{view.channel}] clicked pixel: ({x}, {y})")

        return _cb

    # ---------------------------------------------------------------- helpers

    def _parse_scene_roi_for(self, channel: str):
        """Resolve the scene ROI for one channel from config.

        scene_roi may be:
          - null                       -> interactive per view
          - "x,y,w,h"                  -> same ROI for every view
          - {channel: "x,y,w,h", ...}  -> per-channel ROI
        """
        cfg = self._scene_roi_cfg
        if cfg is None:
            return None
        if isinstance(cfg, (dict,)) or hasattr(cfg, "get"):
            s = cfg.get(channel, None)
        else:
            s = cfg
        if s is None:
            return None
        parts = [int(x) for x in str(s).split(",")]
        if len(parts) != 4:
            raise ValueError(f"scene_roi must be 'x,y,w,h', got {s!r}")
        return tuple(parts)

    @staticmethod
    def _crop_rgbd(rgb, depth, roi):
        sx, sy, sw, sh = roi
        return rgb[sy:sy + sh, sx:sx + sw], depth[sy:sy + sh, sx:sx + sw]

    @staticmethod
    def _crop_K(K, sx, sy):
        Kc = K.copy()
        Kc[0, 2] -= sx
        Kc[1, 2] -= sy
        return Kc

    def _resize_mask_if_needed(self, mask, target_hw):
        h, w = target_hw
        if mask.shape == (h, w):
            return mask
        return cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    def _select_roi_interactive(self, view: _ViewState, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        r = cv2.selectROI(view.window_name, bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(view.window_name)
        # recreate window + callback (selectROI destroys it)
        cv2.namedWindow(view.window_name)
        cv2.setMouseCallback(view.window_name, self._make_mouse_callback(view))
        x, y, w, h = [int(v) for v in r]
        if w == 0 or h == 0:
            # empty selection -> full frame
            H, W = rgb.shape[:2]
            return (0, 0, W, H)
        return (x, y, w, h)

    def _run_sam_prompt(self, view: _ViewState, rgb, point, reload_first_frame):
        torch.set_default_tensor_type("torch.FloatTensor")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if reload_first_frame:
                view.sam2_predictor.load_first_frame(rgb)
            pts = np.array([[point[0], point[1]]], dtype=np.float32)
            lbl = np.array([1], dtype=np.int32)
            _, _, logits = view.sam2_predictor.add_new_prompt(
                frame_idx=0, obj_id=self._sam2_obj_id,
                points=pts, labels=lbl, clear_old_points=True,
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
            vis, "click: re-mask   r: reset   y/Enter: accept",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    def _sam2_step(self, view: _ViewState, rgb_image):
        torch.set_default_tensor_type("torch.FloatTensor")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, mask_logits = view.sam2_predictor.track(rgb_image)
        mask = (mask_logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(bool)
        return mask

    def _apply_mask(self, rgb, depth, mask):
        rgb_out, depth_out = rgb, depth
        if self._sam2_mask_rgb:
            rgb_out = rgb.copy()
            rgb_out[~mask] = 0
        if self._sam2_mask_depth:
            depth_out = depth.copy()
            depth_out[~mask] = 0
        return rgb_out, depth_out

    def _track_score(self, view: _ViewState, rgb, depth):
        """Re-score the raw centered-mesh pose that track_one just cached in
        estimator.pose_last, using this view's cropped intrinsics."""
        est = view.estimator
        pose_last = getattr(est, "pose_last", None)
        if pose_last is None:
            return None
        ob_in_cams = pose_last.reshape(1, 4, 4).data.cpu().numpy()
        scores, _ = est.scorer.predict(
            mesh=est.mesh,
            rgb=rgb,
            depth=depth,
            K=view.K_cropped,
            ob_in_cams=ob_in_cams,
            normal_map=None,
            mesh_tensors=est.mesh_tensors,
            gaussian_tensors=est.gaussian_tensors,
            glctx=est.glctx,
            mesh_diameter=est.diameter,
            get_vis=False,
        )
        return float(scores[0])

    def _is_lost(self, score):
        """True if `score` indicates a lost track.

        Lost detection is INDEPENDENT of the winner-selection direction: even
        when higher score = better fit, an outlier (occlusion / drift) makes the
        re-scored track_one pose "blow up" past ~100. So by default lost means
        score is ABOVE the threshold (lost_side == "above"). Set lost_side to
        "below" only if your scorer instead drops on outliers.
        """
        if self._lost_threshold is None or score is None:
            return False
        if self._lost_side == "below":
            return score < self._lost_threshold
        return score > self._lost_threshold

    def _better(self, a, b):
        """True if score a is strictly better than score b, per direction."""
        if self._score_direction == "max":
            return a > b
        return a < b

    def _margin(self, best, second):
        """Signed margin by which `best` beats `second` (>=0 means best wins)."""
        if self._score_direction == "max":
            return best - second
        return second - best

    # ---------------------------------------------------------------- process

    def _process(self):
        # ---- registration phase: drive each unregistered view's click/preview ----
        if not self._initialized:
            for v in self._views:
                self._process_register(v)
            if all(v.registered for v in self._views):
                self._initialized = True
                if self._use_filter and self._pose_filter is not None:
                    self._pose_filter.reset()
                # flush stale frames / publisher
                for v in self._views:
                    self._drain(self.extr_sub_que_dict[v.channel])
                if self._pose_pub_channel in self.percep_pub_que_dict and \
                        self.percep_pub_que_dict[self._pose_pub_channel]:
                    self._drain(self.percep_pub_que_dict[self._pose_pub_channel])
            return

        # ---- tracking phase: each view tracks + scores; then fuse ----
        any_tracked = False
        for v in self._views:
            if self._track_view(v):
                any_tracked = True
        if not any_tracked:
            return
        self._report_fps()

        # A view is a trustworthy candidate if it produced a pose and is not lost.
        candidates = [
            v for v in self._views
            if v.pose is not None and v.score is not None and not self._is_lost(v.score)
        ]

        # Auto re-register fallback: when no view is trustworthy for several
        # consecutive frames (e.g. the object moved too fast for every tracker),
        # re-register each view from its current SAM2 mask.
        if not candidates:
            self._all_lost_streak += 1
            if self._auto_reset and self._all_lost_streak >= self._auto_reset_streak:
                self._auto_reregister()
                self._all_lost_streak = 0
                # recompute after re-register so a freshly-registered view can win
                candidates = [
                    v for v in self._views
                    if v.pose is not None and v.score is not None
                    and not self._is_lost(v.score)
                ]
        else:
            self._all_lost_streak = 0

        winner = self._select_winner(candidates)
        if winner is None:
            return

        # Only re-sync when there was a trustworthy winner (not a lost fallback).
        if self._resync_enable and candidates:
            self._resync_losers(winner)

        self._publish_and_visualize(winner)

    @staticmethod
    def _drain(q):
        while not q.empty():
            q.get()

    def _pop_latest(self, view: _ViewState):
        """Pop all queued RGBD frames for a view, return the latest (or None)."""
        q = self.extr_sub_que_dict[view.channel]
        if q.empty():
            return None
        rgbd_data = None
        while not q.empty():
            rgbd_data = q.get()
        return rgbd_data

    def _ensure_K_cropped(self, view: _ViewState):
        if view.K_cropped is None:
            sx, sy = view.scene_roi[0], view.scene_roi[1]
            view.K_cropped = self._crop_K(view.cam_intrinsic, sx, sy)

    # ---- registration ----

    def _process_register(self, view: _ViewState):
        if view.registered:
            return
        rgbd_data = self._pop_latest(view)
        if rgbd_data is None:
            return
        rgb_full = rgbd_data.get_rgb_image()
        depth_full = rgbd_data.get_depth_image().astype(np.float32) / view.depth_factor

        if view.init_phase == "need_roi":
            view.scene_roi = self._select_roi_interactive(view, rgb_full)
            self._ensure_K_cropped(view)
            view.init_phase = "wait_click"
            return

        self._ensure_K_cropped(view)
        rgb_crop, depth_crop = self._crop_rgbd(rgb_full, depth_full, view.scene_roi)

        if view.init_phase == "wait_click":
            self._vis_image(rgb_crop.copy(), view.window_name)
            if view.clicked_point is None:
                return
            pt = view.clicked_point
            view.clicked_point = None
            view.preview_rgb = rgb_crop.copy()
            view.preview_depth = depth_crop.copy()
            view.preview_point = pt
            view.preview_mask = self._run_sam_prompt(
                view, view.preview_rgb, pt, reload_first_frame=True
            )
            view.init_phase = "preview"
            return

        if view.init_phase == "preview":
            vis = self._make_preview_vis(
                view.preview_rgb, view.preview_mask, view.preview_point
            )
            self._vis_image(vis, view.window_name)

            if view.clicked_point is not None:
                pt = view.clicked_point
                view.clicked_point = None
                view.preview_point = pt
                view.preview_mask = self._run_sam_prompt(
                    view, view.preview_rgb, pt, reload_first_frame=False
                )
                return

            key = view.last_key
            if key == ord("r"):
                view.init_phase = "wait_click"
                view.preview_rgb = None
                view.preview_depth = None
                view.preview_mask = None
                view.preview_point = None
                view.last_key = -1
                return
            if key in (ord("y"), 13, 10, 32):  # y / Enter / LF / Space
                view.last_key = -1
                view.estimator.register(
                    K=view.K_cropped,
                    rgb=view.preview_rgb,
                    depth=view.preview_depth,
                    ob_mask=view.preview_mask,
                    iteration=self._est_refine_iter,
                )
                view.registered = True
                print(f"[{view.channel}] registered.")
            return

    # ---- tracking ----

    def _track_view(self, view: _ViewState):
        """Track one frame for a view. Updates view.pose / view.score.
        Returns True if a pose was produced this frame."""
        rgbd_data = self._pop_latest(view)
        if rgbd_data is None:
            return False
        view.last_rgbd_time = rgbd_data.get_time()
        rgb_full = rgbd_data.get_rgb_image()
        depth_full = rgbd_data.get_depth_image().astype(np.float32) / view.depth_factor

        rgb_crop, depth_crop = self._crop_rgbd(rgb_full, depth_full, view.scene_roi)

        with self._stage_timer("sam2"):
            mask = self._sam2_step(view, rgb_crop)
        mask = self._resize_mask_if_needed(mask, rgb_crop.shape[:2])
        rgb_in, depth_in = self._apply_mask(rgb_crop, depth_crop, mask)

        with self._stage_timer("track"):
            pose = view.estimator.track_one(
                rgb=rgb_in,
                depth=depth_in,
                K=view.K_cropped,
                iteration=self._track_refine_iter,
            )
        view.pose = pose  # ob_in_cam at oriented-bbox origin
        with self._stage_timer("score"):
            view.score = self._track_score(view, rgb_in, depth_in)

        # stash for visualization and for auto re-register fallback
        view._rgb_crop = rgb_crop
        view._depth_crop = depth_crop
        view._mask = mask
        return pose is not None

    # ---- fusion ----

    def _view_world_pose_centered(self, view: _ViewState):
        """Lift this view's raw pose_last (centered-mesh, view-cam frame) to
        world frame. Returns 4x4 world pose of the centered mesh, or None."""
        pose_last = getattr(view.estimator, "pose_last", None)
        if pose_last is None:
            return None
        p = pose_last.data.cpu().numpy().reshape(4, 4)
        return view.cam2world @ p

    def _select_winner(self, candidates):
        """Pick the best-scoring, non-lost view from `candidates`. Falls back to
        any view with a pose (so tracking keeps publishing) when none are
        trustworthy. Returns a _ViewState or None."""
        if not candidates:
            fallback = [v for v in self._views if v.pose is not None]
            return fallback[0] if fallback else None

        winner = candidates[0]
        for v in candidates[1:]:
            if self._better(v.score, winner.score):
                winner = v
        winner._candidates = candidates
        return winner

    def _auto_reregister(self):
        """Re-register every view from its current SAM2 mask. Skips a view whose
        SAM2 mask is empty (SAM2 itself lost the object, e.g. moved out of view)
        or whose latest frame is missing."""
        for v in self._views:
            mask = getattr(v, "_mask", None)
            rgb = getattr(v, "_rgb_crop", None)
            depth = getattr(v, "_depth_crop", None)
            if mask is None or rgb is None or depth is None:
                continue
            if int(mask.sum()) < 4:
                logging.warning(f"[{v.channel}] auto-reregister skipped: empty mask")
                continue
            logging.warning(f"[{v.channel}] auto-reregister from SAM2 mask")
            v.pose = v.estimator.register(
                K=v.K_cropped,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=self._est_refine_iter,
            )
            v.score = self._track_score(v, rgb, depth)

    def _resync_losers(self, winner: _ViewState):
        """Overwrite losing views' pose_last with the winner's pose, expressed in
        each loser's camera frame. Only applies when the winner beats the
        runner-up by at least resync_margin (cross-view score guard)."""
        candidates = getattr(winner, "_candidates", None)
        if not candidates or len(candidates) < 2:
            # only one trustworthy view; nothing reliable to sync from/to,
            # but we can still pull clearly-lost views back to the winner below.
            pass
        else:
            others = [c for c in candidates if c is not winner]
            best_other = others[0]
            for c in others[1:]:
                if self._better(c.score, best_other.score):
                    best_other = c
            if self._margin(winner.score, best_other.score) < self._resync_margin:
                return  # not confident enough to overwrite

        world_centered = self._view_world_pose_centered(winner)
        if world_centered is None:
            return

        for v in self._views:
            if v is winner or v.estimator is None:
                continue
            # push winner's world pose into loser v's camera frame
            pose_last_v = v.world2cam @ world_centered
            v.estimator.pose_last = torch.as_tensor(
                pose_last_v, device="cuda", dtype=torch.float
            )

    # ---- output ----

    def _publish_and_visualize(self, winner: _ViewState):
        # winner.pose is ob_in_cam at oriented-bbox origin, in winner camera frame
        pose = winner.pose
        center_pose_cam = pose @ self._to_origin_inv
        pose_world = winner.cam2world @ center_pose_cam

        if self._use_filter and self._pose_filter is not None:
            pose_world = self._pose_filter.update(pose_world)

        if self._fix_x_rotation:
            x_axis = pose_world[:3, 0]
            y_axis = np.cross(x_axis, [0, 1, 0]); y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis)
            pose_world[:3, 0] = x_axis
            pose_world[:3, 1] = y_axis
            pose_world[:3, 2] = z_axis
        if self._fix_z_rotation:
            z_axis = pose_world[:3, 2]
            x_axis = np.cross([1, 0, 0], z_axis); x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
            pose_world[:3, 0] = x_axis
            pose_world[:3, 1] = y_axis
            pose_world[:3, 2] = z_axis

        self._pose_world = pose_world

        if self._publisher_name == "named_vec":
            pose_out = np.concatenate(
                [pose_world[:3, 3], pose_world[:3, :3].flatten()], axis=0
            ).reshape(1, 12)
            out_data = NamedVecListData(
                num_vecs=1, vec_dim=12, name=self._object_name
            )
            out_data.set_data(
                winner.last_rgbd_time, [self._object_name], pose_out
            )
            if self.percep_pub_que_dict.get(self._pose_pub_channel):
                self.percep_pub_que_dict[self._pose_pub_channel].put(out_data)

        if self._visualize:
            for v in self._views:
                if v.pose is None or getattr(v, "_rgb_crop", None) is None:
                    continue
                center_pose_v = v.pose @ self._to_origin_inv
                rgb_vis = v._rgb_crop.copy()
                color = (0, 255, 0) if v is winner else (0, 165, 255)
                draw_posed_3d_box(
                    v.K_cropped, rgb_vis, center_pose_v, self._bbox, line_color=color
                )
                rgb_vis = draw_xyz_axis(
                    rgb_vis, ob_in_cam=center_pose_v, K=v.K_cropped,
                    thickness=3, transparency=0, is_input_rgb=True,
                )
                tag = "WINNER" if v is winner else "sync"
                score_txt = "-" if v.score is None else f"{v.score:.1f}"
                cv2.putText(
                    rgb_vis, f"{tag} score={score_txt}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )
                self._vis_image(rgb_vis, v.window_name)

        # Live render overlay window(s): the refiner's rendered model (col A)
        # beside the observed crop (col B), plus depth panels, for every view
        # that produced a vis this tick. Separate window so it doesn't clobber
        # the bbox/axis overlay above.
        if self._visualize_render:
            for v in self._views:
                render_vis = getattr(v, "_render_vis", None)
                if render_vis is not None:
                    self._vis_image(render_vis, f"{v.window_name} [render]")

    # ---- gui ----

    def _vis_image(self, image, window_name):
        if image.shape[-1] == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image
        cv2.imshow(window_name, bgr_image)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            # route key to whichever view owns this window
            for v in self._views:
                if v.window_name == window_name:
                    v.last_key = key
                    break
