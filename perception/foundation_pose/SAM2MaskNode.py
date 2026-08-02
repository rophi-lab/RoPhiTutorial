"""
SAM2 mask-streaming node for the split multi-view FoundationPose pipeline.

Runs streaming SAM2 per camera view and publishes a single-object mask per frame
on an LCM channel, each mask stamped with the SOURCE RGB FRAME's timestamp and a
view id. A separate FoundationPose node subscribes to these masks and time-
matches them against its own buffered RGBD frames to (re)register the object
pose. Splitting SAM2 (this node) from FoundationPose (the other node) lets the
two run concurrently in separate processes/GPUs instead of serially.

Lifecycle per view: draw a scene ROI (or take it from config) -> click the object
-> preview/accept the first mask -> stream masks every frame. Masking of the
object out of the RGB/depth is intentionally NOT done here; the FP node runs
track_one on the raw crop and only uses masks at (re)registration time.
"""

from typing import List, Optional
import logging
import os
import time

import numpy as np
import torch

from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_camera_predictor

from perception.BasePerception import BasePerception
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.MaskData import MaskData

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV (cv2) is required for SAM2MaskNode. Install opencv-python."
    ) from e


class _ViewState:
    """Per-view SAM2 streaming + registration state."""

    def __init__(self, channel: str, view_id: int, mask_channel: str):
        self.channel = channel
        self.view_id = view_id
        self.mask_channel = mask_channel
        self.window_name = f"SAM2MaskNode [{channel}]"

        self.sam2_predictor = None
        self.depth_factor: Optional[float] = None  # only for consistency; unused

        # register / preview state machine. SAM2 runs on the FULL frame here;
        # ROI cropping is done only in the FP node.
        self.init_phase = "wait_click"  # wait_click -> preview -> streaming
        self.streaming = False
        self.clicked_point: Optional[tuple] = None
        self.last_key = -1
        self.preview_rgb: Optional[np.ndarray] = None
        self.preview_point: Optional[tuple] = None
        self.preview_mask: Optional[np.ndarray] = None

        # last wall-clock time SAM2 tracked this view (for publish-rate throttle).
        self.last_track_time: Optional[float] = None


class SAM2MaskNode(BasePerception):
    """Streams SAM2 masks (one channel per view) for the split FP pipeline."""

    def __init__(self, config):
        super().__init__(config)

        self._debug = config.get("debug", 0)
        torch.backends.cudnn.benchmark = True

        # Publish-rate throttle: run SAM2 (and publish a mask) at most this many
        # times per second per view. Masks feed the FP node's re-registration
        # only, so a low rate saves GPU/power; the FP node time-matches masks
        # against its own RGBD ring buffer regardless. <= 0 disables throttling.
        self._sam2_publish_freq = float(config.get("publish_freq", 2.0))
        self._sam2_min_dt = (
            1.0 / self._sam2_publish_freq if self._sam2_publish_freq > 0 else 0.0
        )

        # ---- SAM2 config ----
        sam2_cfg = config.get("sam2", {})
        self._sam2_config_file = sam2_cfg.get(
            "config_file", "configs/sam2.1/sam2.1_hiera_t.yaml"
        )
        self._sam2_hf_repo = sam2_cfg.get("hf_repo", "facebook/sam2.1-hiera-tiny")
        self._sam2_hf_filename = sam2_cfg.get("hf_filename", "sam2.1_hiera_tiny.pt")
        self._sam2_checkpoint = sam2_cfg.get("checkpoint", None)
        self._sam2_obj_id = sam2_cfg.get("obj_id", 1)

        compile_cfg = sam2_cfg.get("compile", {})
        self._compile_image_encoder = compile_cfg.get("image_encoder", False)

        # ---- channels ----
        sub_cfg = config.get("sub_manager", {})
        channels = sub_cfg.get("rgbd_channels", None)
        if channels is None:
            channels = [sub_cfg.get("rgbd_channel", "d455_1")]
        self._channels: List[str] = list(channels)

        # per-view mask output channels. Config may give an explicit list;
        # otherwise default to "<rgbd_channel>_mask".
        pub_cfg = config["pub_manager"]
        mask_channels = pub_cfg.get("mask_channels", None)
        if mask_channels is None:
            mask_channels = [ch + "_mask" for ch in self._channels]
        if len(mask_channels) != len(self._channels):
            raise ValueError(
                f"mask_channels ({len(mask_channels)}) must match rgbd_channels "
                f"({len(self._channels)})"
            )
        self._mask_channels = list(mask_channels)

        self._views: List[_ViewState] = [
            _ViewState(ch, i, self._mask_channels[i])
            for i, ch in enumerate(self._channels)
        ]

        self._visualize = config.get("visualize", True)

    # ------------------------------------------------------------------ init

    def initialize(self):
        for v in self._views:
            cv2.namedWindow(v.window_name)
            cv2.setMouseCallback(v.window_name, self._make_mouse_callback(v))

        if self._sam2_checkpoint and os.path.isfile(self._sam2_checkpoint):
            ckpt_path = self._sam2_checkpoint
        else:
            ckpt_path = hf_hub_download(
                repo_id=self._sam2_hf_repo, filename=self._sam2_hf_filename
            )
        logging.info(f"SAM2 checkpoint: {ckpt_path}")

        overrides = []
        if self._compile_image_encoder:
            overrides.append("++model.compile_image_encoder=true")

        for v in self._views:
            v.sam2_predictor = build_sam2_camera_predictor(
                self._sam2_config_file, ckpt_path, device="cuda",
                vos_optimized=True, hydra_overrides_extra=overrides,
            )
            v.init_phase = "wait_click"

        # wait for each view's camera info (only depth_factor is relevant here,
        # but consuming the queue keeps it from backing up).
        pending = {v.channel: v for v in self._views}
        while pending:
            for ch, v in list(pending.items()):
                info_q = self.extr_sub_que_dict[ch + "_info"]
                if not info_q.empty():
                    cam_info = info_q.get()
                    if isinstance(cam_info, CameraInfoData):
                        v.depth_factor = cam_info.get_depth_factor()
                        print(f"[{ch}] camera info received.")
                        del pending[ch]

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    # ---------------------------------------------------------------- helpers

    def _make_mouse_callback(self, view: _ViewState):
        def _cb(event, x, y, flags, param=None):
            if event == cv2.EVENT_LBUTTONDOWN:
                view.clicked_point = (x, y)
                print(f"[{view.channel}] clicked pixel: ({x}, {y})")
        return _cb

    def _resize_mask_if_needed(self, mask, target_hw):
        h, w = target_hw
        if mask.shape == (h, w):
            return mask
        return cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

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

    def _sam2_step(self, view: _ViewState, rgb_image):
        torch.set_default_tensor_type("torch.FloatTensor")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, mask_logits = view.sam2_predictor.track(rgb_image)
        mask = (mask_logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(bool)
        return mask

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

    def _vis_image(self, image, window_name):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image
        cv2.imshow(window_name, bgr)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            for v in self._views:
                if v.window_name == window_name:
                    v.last_key = key
                    break

    def _pop_latest(self, view: _ViewState):
        q = self.extr_sub_que_dict[view.channel]
        if q.empty():
            return None
        data = None
        while not q.empty():
            data = q.get()
        return data

    def _publish_mask(self, view: _ViewState, mask, t):
        """Publish a bool/uint8 mask stamped with the source rgb timestamp."""
        mask_u8 = mask.astype(np.uint8)
        out = MaskData(
            height=mask_u8.shape[0], width=mask_u8.shape[1],
            channel_type=1, view_id=view.view_id, name=view.mask_channel,
        )
        out.set_data(t=t, view_id=view.view_id, mask_image=mask_u8, channel_type=1)
        q = self.percep_pub_que_dict.get(view.mask_channel)
        if q is not None:
            q.put(out)

    # ---------------------------------------------------------------- process

    def _process(self):
        for v in self._views:
            self._process_view(v)

    def _process_view(self, view: _ViewState):
        rgbd_data = self._pop_latest(view)
        if rgbd_data is None:
            return
        rgb_full = rgbd_data.get_rgb_image()
        t = rgbd_data.get_time()

        # SAM2 runs on the FULL frame; ROI cropping is done only in the FP node.
        # ---- registration (click -> preview -> accept) ----
        if not view.streaming:
            if view.init_phase == "wait_click":
                self._vis_image(rgb_full.copy(), view.window_name)
                if view.clicked_point is None:
                    return
                pt = view.clicked_point
                view.clicked_point = None
                view.preview_rgb = rgb_full.copy()
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
                    view.preview_mask = None
                    view.preview_point = None
                    view.last_key = -1
                    return
                if key in (ord("y"), 13, 10, 32):
                    view.last_key = -1
                    view.streaming = True
                    print(f"[{view.channel}] streaming masks -> {view.mask_channel}")
                return

        # ---- streaming: track SAM2 on the full frame, publish the mask ----
        # Publish-rate throttle: skip this frame (its rgbd was already drained by
        # _pop_latest) if we tracked too recently. Advancing SAM2 less often is
        # fine here -- masks feed re-registration only, not per-frame tracking --
        # and it frees the GPU for the FP node. Fast motion may cost some mask
        # continuity (SAM2 memory window widens in wall-clock time).
        now = time.time()
        if (
            self._sam2_min_dt > 0.0
            and view.last_track_time is not None
            and (now - view.last_track_time) < self._sam2_min_dt
        ):
            return
        view.last_track_time = now

        mask = self._sam2_step(view, rgb_full)
        mask = self._resize_mask_if_needed(mask, rgb_full.shape[:2])
        self._publish_mask(view, mask, t)

        if self._visualize:
            bgr = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)
            overlay = bgr.copy()
            overlay[mask] = (0, 255, 255)
            vis = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
            cv2.imshow(view.window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                view.last_key = key
