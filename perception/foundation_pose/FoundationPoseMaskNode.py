"""
FoundationPose node for the split multi-view pipeline.

Subscribes to RGBD (per view) AND to SAM2 masks (per view) produced by a separate
SAM2MaskNode process. It runs FoundationPose track_one every frame on the raw crop
(no SAM2 inline), and (re)registers by TIME-MATCHING a received mask against a ring
buffer of recent RGBD frames: the mask carries the timestamp of the RGB frame it
was computed on, so the matching frame's rgb/depth is used for register().

Reuses FoundationPoseTrackingMultiView for all the shared fusion/scoring/publish
logic; only the SAM2-touching parts (init, registration, per-frame track) are
overridden. mask_rgb/mask_depth are irrelevant here (track_one runs on the raw
crop); masks are used for registration only.
"""

from typing import List, Optional
from collections import deque
import logging

import numpy as np

from perception.foundation_pose.FoundationPoseTrackingMultiView import (
    FoundationPoseTrackingMultiView,
    _ViewState,
)
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.MaskData import MaskData
from utils.lie.se3 import invSE3

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV (cv2) is required for FoundationPoseMaskNode.") from e


class FoundationPoseMaskNode(FoundationPoseTrackingMultiView):
    """Split-pipeline FoundationPose node: masks arrive over LCM, not from an
    inline SAM2."""

    def __init__(self, config):
        # Base __init__ builds SAM2-related config we don't need, but it is
        # harmless (no predictor is created until initialize, which we override).
        super().__init__(config)

        # mask input channels, aligned 1:1 with rgbd channels / views.
        sub_cfg = config.get("sub_manager", {})
        mask_channels = sub_cfg.get("mask_channels", None)
        if mask_channels is None:
            mask_channels = [ch + "_mask" for ch in self._channels]
        if len(mask_channels) != len(self._views):
            raise ValueError(
                f"mask_channels ({len(mask_channels)}) must match rgbd_channels "
                f"({len(self._views)})"
            )
        self._mask_channels = list(mask_channels)
        for v, mc in zip(self._views, self._mask_channels):
            v.mask_channel = mc
            # ring buffer of recent (timestamp, rgb_crop, depth_crop) for
            # time-matched registration.
            v.rgbd_ring = deque(maxlen=int(config.get("rgbd_buffer_size", 60)))
            # latest mask received for this view (MaskData) and whether we have
            # completed the initial registration.
            v.latest_mask: Optional[MaskData] = None
            v.registered = False

        # how close (seconds) a buffered frame must be to the mask timestamp.
        self._mask_match_tol = float(config.get("mask_match_tolerance", 0.2))

        # When scene_roi is null: True -> draw an ROI per view interactively on
        # the first frame; False -> use the full frame. A fixed "x,y,w,h" in
        # scene_roi always wins regardless of this flag.
        self._interactive_roi = bool(config.get("interactive_roi", True))

    # ------------------------------------------------------------------ init

    def initialize(self):
        # No SAM2 here. Build one FoundationPose estimator per view (sharing
        # scorer/refiner/glctx) and wait for camera parameters.
        from third_party.FoundationPose.estimater import FoundationPose

        for v in self._views:
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
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # Skip FoundationPoseTrackingMultiView.stop's SAM2 GUI teardown path is
        # already covered; call the grandparent (BasePerception) stop directly.
        from perception.BasePerception import BasePerception
        BasePerception.stop(self)

    # ---------------------------------------------------------------- helpers

    def _drain_masks(self, view: _ViewState):
        """Consume the mask channel, keeping only the latest mask for a view.
        The scene_roi crop is applied here so masks align with the FP crop; the
        SAM2 node already crops with the same ROI, so masks arrive cropped."""
        q = self.extr_sub_que_dict.get(view.mask_channel)
        if q is None:
            return
        while not q.empty():
            md = q.get()
            if isinstance(md, MaskData):
                view.latest_mask = md

    def _match_frame(self, view: _ViewState, t_mask):
        """Return (rgb, depth) from the ring buffer whose timestamp is closest to
        t_mask within tolerance, else (None, None)."""
        best = None
        best_dt = None
        for t, rgb, depth in list(view.rgbd_ring):
            dt = abs(t - t_mask)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = (rgb, depth)
        if best is None or best_dt > self._mask_match_tol:
            return None, None
        return best

    @staticmethod
    def _crop_mask(mask, roi):
        """Crop a full-frame mask with an (x, y, w, h) ROI. roi None -> no crop."""
        if roi is None:
            return mask
        sx, sy, sw, sh = roi
        return mask[sy:sy + sh, sx:sx + sw]

    def _resolve_roi(self, view: _ViewState, rgb_full):
        """Resolve this view's ROI on the first frame: draw it interactively when
        interactive_roi is on, else use the full frame. Returns (x, y, w, h)."""
        h, w = rgb_full.shape[:2]
        if not self._interactive_roi:
            return (0, 0, w, h)
        win = f"FoundationPoseMaskNode ROI [{view.channel}]"
        bgr = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)
        print(f"[{view.channel}] draw ROI then press ENTER (or ENTER for full frame)")
        r = cv2.selectROI(win, bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)
        x, y, rw, rh = [int(val) for val in r]
        if rw == 0 or rh == 0:
            return (0, 0, w, h)
        return (x, y, rw, rh)

    def _register_from_mask(self, view: _ViewState):
        """Try to (re)register this view from its latest mask, time-matched to a
        buffered RGBD frame. Returns True on success."""
        md = view.latest_mask
        if md is None:
            return False
        t_mask, _, mask = md.get_data()
        mask = mask.astype(bool)
        # The SAM2 node publishes a FULL-frame mask; crop it with this view's ROI
        # so it aligns with the cropped rgb/depth (which the ring buffer stores).
        mask = self._crop_mask(mask, view.scene_roi)
        if int(mask.sum()) < 4:
            return False
        rgb, depth = self._match_frame(view, t_mask)
        if rgb is None:
            return False
        # guard against off-by-one crop size differences.
        mask = self._resize_mask_if_needed(mask, rgb.shape[:2])
        self._ensure_K_cropped(view)
        view.pose = view.estimator.register(
            K=view.K_cropped, rgb=rgb, depth=depth,
            ob_mask=mask, iteration=self._est_refine_iter,
        )
        view.score = self._track_score(view, rgb, depth)
        return True

    # ---------------------------------------------------------------- process

    def _process(self):
        # Pull RGBD for every view into its ring buffer, and drain masks.
        # _got_new gates the expensive track/score/publish path below: without
        # it we re-ran track_one (+ score + gsplat render + viz) on the SAME
        # cached crop every loop iteration (~kHz busy loop), pegging CPU/GPU
        # while the camera only delivers ~30 Hz. Track only on fresh frames.
        any_new_frame = False
        for v in self._views:
            rgbd = self._pop_latest(v)
            v._got_new = rgbd is not None
            if rgbd is not None:
                any_new_frame = True
                rgb_full = rgbd.get_rgb_image()
                depth_full = (
                    rgbd.get_depth_image().astype(np.float32) / v.depth_factor
                )
                # Resolve a null scene_roi once we know the frame size. When
                # interactive_roi is on, block on cv2.selectROI per view; else
                # fall back to the full frame.
                if v.scene_roi is None:
                    v.scene_roi = self._resolve_roi(v, rgb_full)
                self._ensure_K_cropped(v)
                rgb_crop, depth_crop = self._crop_rgbd(
                    rgb_full, depth_full, v.scene_roi
                )
                v.rgbd_ring.append((rgbd.get_time(), rgb_crop, depth_crop))
                v.last_rgbd_time = rgbd.get_time()
                v._latest_crop = (rgb_crop, depth_crop)
            self._drain_masks(v)

        # ---- registration phase: register each view once from its first mask --
        if not self._initialized:
            for v in self._views:
                if not v.registered:
                    if self._register_from_mask(v):
                        v.registered = True
                        print(f"[{v.channel}] registered from mask.")
            if all(v.registered for v in self._views):
                self._initialized = True
                if self._use_filter and self._pose_filter is not None:
                    self._pose_filter.reset()
            return

        # ---- tracking phase: track_one on raw crop (no SAM2) ----
        # Nothing new arrived this tick -> the pose can't have changed, so skip
        # the whole track/score/select/publish path and let the loop idle.
        if not any_new_frame:
            return
        any_tracked = False
        for v in self._views:
            # Only re-track views that received a fresh frame; others keep their
            # last pose/score so multi-view winner selection still sees them.
            if v._got_new and self._track_view_maskless(v):
                any_tracked = True
        if not any_tracked:
            return
        self._report_fps()

        candidates = [
            v for v in self._views
            if v.pose is not None and v.score is not None and not self._is_lost(v.score)
        ]

        if not candidates:
            self._all_lost_streak += 1
            if self._auto_reset and self._all_lost_streak >= self._auto_reset_streak:
                self._auto_reregister()
                self._all_lost_streak = 0
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
        if self._resync_enable and candidates:
            self._resync_losers(winner)
        self._publish_and_visualize(winner)

    def _track_view_maskless(self, view: _ViewState):
        """track_one + score on the latest raw crop (no SAM2, no masking)."""
        crop = getattr(view, "_latest_crop", None)
        if crop is None:
            return False
        rgb_in, depth_in = crop
        # Live render overlay: track_one only fills extra['vis'] when the
        # estimator's debug level is >= 2 (it computes the rendered-vs-observed
        # canvas but does NO file I/O there, unlike register). Bump debug just
        # for this call so we get the vis without turning on the heavy
        # register-time debug dumps, then restore it.
        extra = {}
        _dbg_saved = view.estimator.debug
        if self._visualize_render:
            view.estimator.debug = max(_dbg_saved, 2)
        try:
            with self._stage_timer("track"):
                pose = view.estimator.track_one(
                    rgb=rgb_in, depth=depth_in, K=view.K_cropped,
                    iteration=self._track_refine_iter,
                    extra=extra,
                )
        finally:
            view.estimator.debug = _dbg_saved
        view.pose = pose
        # Rendered model (col A) vs observed crop (col B); None when the flag is
        # off. Displayed by _publish_and_visualize.
        view._render_vis = extra.get("vis", None)
        with self._stage_timer("score"):
            view.score = self._track_score(view, rgb_in, depth_in)
        view._rgb_crop = rgb_in  # for visualization
        return pose is not None

    def _auto_reregister(self):
        """Re-register every view from its most recent received mask, time-matched
        to a buffered RGBD frame. Skips views with no usable mask."""
        for v in self._views:
            if self._register_from_mask(v):
                logging.warning(f"[{v.channel}] auto-reregister from LCM mask")
