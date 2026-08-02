"""Differentiable base–camera extrinsic calibration (SAM2 + nvdiffrast)."""

from __future__ import annotations

import signal
import time
from queue import Empty, Queue
from typing import List, Optional, Tuple

import cv2
import lcm
import numpy as np
import torch
from omegaconf import DictConfig

from communication.lcm.subscriber.data_subscriber.CameraInfoSubscriber import (
    CameraInfoSubscriber,
)
from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)
from communication.lcm.subscriber.data_subscriber.RGBDSubscriber import RGBDSubscriber
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.JointMeasData import JointMeasData
from data_type.basic_types.RGBDData import RGBDData
from utils.calibration.diff_silhouette import (
    DiffSilhouetteRenderer,
    se3_from_wt,
    wt_from_se3,
)
from utils.calibration.robot_mesh_fk import RobotMeshFK
from utils.lie.se3 import invSE3


def _rgb_to_gray01(rgb: np.ndarray) -> np.ndarray:
    g = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    return np.clip(g / 255.0, 0.0, 1.0)


def _perturb_cam_in_base(
    T_cam2base: np.ndarray, rot_deg: float, trans_m: float, rng: np.random.Generator
):
    """Left-multiply a small SE(3) on cam-in-base (OpenCV eye / axes in world).

    Same convention as Manipulator-Software eyeball_camera_pose / EasyHeC:
    perturbing here moves/rotates the camera in the arm-base frame. Do **not**
    left-multiply noise onto T_world2cam — that warps the pose badly.
    """
    from utils.lie.so3 import exp_so3

    w = rng.normal(size=3)
    w = w / (np.linalg.norm(w) + 1e-9) * np.deg2rad(rot_deg)
    t = rng.normal(size=3)
    t = t / (np.linalg.norm(t) + 1e-9) * trans_m
    dT = np.eye(4, dtype=np.float64)
    dT[:3, :3] = exp_so3(w)
    dT[:3, 3] = t
    return (dT @ np.asarray(T_cam2base, dtype=np.float64)).astype(np.float32)


def _pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    """Translation (m) and rotation (deg) of T_est relative to T_gt (same SE3 frame)."""
    from utils.lie.so3 import log_SO3, unskew

    dT = T_est @ invSE3(T_gt)
    t_err = float(np.linalg.norm(dT[:3, 3]))
    w = unskew(log_SO3(dT[:3, :3]))
    r_err = float(np.rad2deg(np.linalg.norm(w)))
    return t_err, r_err


class CalibInterrupted(KeyboardInterrupt):
    """Raised when the user requests stop (SIGINT/SIGTERM or q in a window)."""


class DiffBaseToCamCalibrator:
    """LCM capture → SAM2 masks → Adam on OpenCV T_cam2base (cam_in_base)."""

    def __init__(self, config: DictConfig):
        self.cfg = config
        self.num_joints = int(config.get("num_joints", 27))
        self.num_arm = int(config.get("num_arm_joints", 7))
        self.num_hand = int(config.get("num_hand_joints", 20))
        self._stop = False
        self._prev_sigint = None
        self._prev_sigterm = None

        self._lcm = lcm.LCM()
        self._rgb_q: Queue = Queue()
        self._info_q: Queue = Queue()
        self._arm_q: Queue = Queue()
        self._hand_q: Queue = Queue()

        cam_ch = config["rgbd_channel"]
        # Keep strong refs — LCM callbacks are bound methods; dropping the
        # subscriber objects can break delivery under GC pressure.
        self._subscribers = [
            RGBDSubscriber(self._lcm, self._rgb_q),
            CameraInfoSubscriber(self._lcm, self._info_q),
            JointMeasSubscriber(self._lcm, self._arm_q, self.num_arm),
            JointMeasSubscriber(self._lcm, self._hand_q, self.num_hand),
        ]
        self._subscribers[0].subscribe(cam_ch)
        self._subscribers[1].subscribe(cam_ch + "_info")
        self._subscribers[2].subscribe(config["arm_joint_meas_channel"])
        self._subscribers[3].subscribe(config["hand_joint_meas_channel"])
        print(
            f"[Calib] Subscribed joints: "
            f"{config['arm_joint_meas_channel']} ({self.num_arm}), "
            f"{config['hand_joint_meas_channel']} ({self.num_hand})"
        )
        # Joint commands come from GravCompController — this script does not publish ctrl.

        exclude_prefixes = config.get("exclude_link_prefixes", None)
        if exclude_prefixes is not None:
            exclude_prefixes = list(exclude_prefixes)
        self._fk = RobotMeshFK(
            urdf_path=config.get(
                "urdf_path", "assets/scene/flexiv_arm/urdf/Rizon4_viser.urdf"
            ),
            max_faces=int(config.get("max_faces", 0)),
            arm_only=bool(config.get("arm_only", True)),
            exclude_link_prefixes=exclude_prefixes,
        )
        self._sam2 = None
        self._renderer: Optional[DiffSilhouetteRenderer] = None
        self._K: Optional[np.ndarray] = None
        # OpenCV conventions (eyeball / EasyHeC): cam_in_base = inv(world2cam).
        self._T_gt_w2c: Optional[np.ndarray] = None
        self._T_gt_c2b: Optional[np.ndarray] = None
        self._H = int(config.get("height", 480))
        self._W = int(config.get("width", 640))
        self._last_rgb: Optional[np.ndarray] = None
        self._last_arm_q: Optional[np.ndarray] = None
        self._last_hand_q: Optional[np.ndarray] = None
        self._last_q: Optional[np.ndarray] = None
        self._joint_update_count = 0

    def _on_signal(self, signum, frame) -> None:
        # Flag only — raising from here is unreliable inside OpenCV / CUDA C calls.
        self._stop = True
        print("\n[Calib] Interrupt received — exiting loops…", flush=True)

    def _install_signal_handlers(self) -> None:
        self._stop = False
        self._prev_sigint = signal.signal(signal.SIGINT, self._on_signal)
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_signal)

    def _restore_signal_handlers(self) -> None:
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)

    def _check_stop(self) -> None:
        if self._stop:
            raise CalibInterrupted("Calibration interrupted by user")

    def _wait_key(self, delay_ms: int = 15) -> int:
        """cv2.waitKey that also honors SIGINT/SIGTERM via self._stop."""
        self._check_stop()
        key = cv2.waitKey(max(1, int(delay_ms))) & 0xFF
        self._check_stop()
        return key

    def _wait_key_until(self, delay_ms: int = 0) -> int:
        """Like waitKey(0), but poll so Ctrl+C can break out of OpenCV."""
        if delay_ms > 0:
            return self._wait_key(delay_ms)
        while True:
            key = self._wait_key(50)
            if key != 255 and key != 0:
                return key

    def _spin_lcm(self, dt: float = 0.01) -> None:
        self._check_stop()
        self._lcm.handle_timeout(int(dt * 1000))

    def _drain(self, q: Queue):
        last = None
        while True:
            try:
                last = q.get_nowait()
            except Empty:
                break
        return last

    def _wait_camera_info(self, timeout: float = 30.0) -> CameraInfoData:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._spin_lcm(0.05)
            info = self._drain(self._info_q)
            if isinstance(info, CameraInfoData):
                return info
        raise TimeoutError("Timed out waiting for CameraInfoData")

    def _poll_sensors(self) -> None:
        """Drain LCM and refresh cached RGB / q.

        Arm and hand are updated independently. Requiring both messages in the
        *same* drain (old behavior) dropped updates whenever one queue was empty
        and made `_last_q` look stuck at the startup pose.
        """
        self._spin_lcm(0.01)
        rgb_data = self._drain(self._rgb_q)
        if isinstance(rgb_data, RGBDData):
            self._last_rgb = rgb_data.get_rgb_image()

        arm = self._drain(self._arm_q)
        hand = self._drain(self._hand_q)
        updated = False
        if isinstance(arm, JointMeasData):
            self._last_arm_q = arm.get_q()
            updated = True
        if isinstance(hand, JointMeasData):
            self._last_hand_q = hand.get_q()
            updated = True
        if self._last_arm_q is not None and self._last_hand_q is not None:
            self._last_q = np.concatenate([self._last_arm_q, self._last_hand_q])
            if updated:
                self._joint_update_count += 1
        elif self._last_arm_q is not None and self.num_hand == 0:
            self._last_q = self._last_arm_q.copy()
            if updated:
                self._joint_update_count += 1

    def _latest_rgb(self) -> Optional[np.ndarray]:
        self._poll_sensors()
        return self._last_rgb

    def _latest_q(self) -> Optional[np.ndarray]:
        self._poll_sensors()
        return self._last_q

    def _init_sam2(self):
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2_camera_predictor

        repo = self.cfg.get("sam2_hf_repo", "facebook/sam2.1-hiera-tiny")
        fname = self.cfg.get("sam2_hf_filename", "sam2.1_hiera_tiny.pt")
        cfg_file = self.cfg.get("sam2_config_file", "configs/sam2.1/sam2.1_hiera_t.yaml")
        ckpt = hf_hub_download(repo_id=repo, filename=fname)
        self._sam2 = build_sam2_camera_predictor(
            cfg_file, ckpt, device="cuda", vos_optimized=True
        )
        print("[Calib] SAM2 ready.")

    def _sam2_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Interactive click segmentation; returns bool mask HxW."""
        if self._sam2 is None:
            self._init_sam2()

        clicks: List[Tuple[int, int]] = []
        win = "Click robot (LMB add, ENTER confirm, R reset)"
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(win, vis)

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        cv2.imshow(win, vis)
        print("[Calib] Click on the robot arm/hand, then press ENTER.")
        while True:
            key = self._wait_key(20)
            if key in (13, 10, 32):  # Enter / Space
                if clicks:
                    break
            if key in (ord("r"), ord("R")):
                clicks.clear()
                vis[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(win, vis)
            if key in (ord("q"), 27):
                cv2.destroyWindow(win)
                raise RuntimeError("SAM2 selection cancelled")
        cv2.destroyWindow(win)

        pts = np.array(clicks, dtype=np.float32)
        lbl = np.ones(len(clicks), dtype=np.int32)
        torch.set_default_tensor_type("torch.FloatTensor")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._sam2.load_first_frame(rgb)
            _, _, logits = self._sam2.add_new_prompt(
                frame_idx=0, obj_id=1, points=pts, labels=lbl
            )
        mask = (logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(bool)
        return mask

    def capture_views(self) -> List[dict]:
        """Live RGB preview; user poses under grav-comp and presses SPACE to capture."""
        info = self._wait_camera_info()
        self._K = info.get_intrinsic().astype(np.float32)
        self._T_gt_w2c = info.get_extrinsic().astype(np.float32)
        self._T_gt_c2b = invSE3(self._T_gt_w2c).astype(np.float32)
        self._H, self._W = info.get_img_dim()
        eye = self._T_gt_c2b[:3, 3]
        print("[Calib] Got K and GT OpenCV extrinsics from CameraInfo.")
        print(
            f"[Calib] GT cam_in_base eye (m) = [{eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f}]"
        )

        n_want = int(self.cfg.get("num_captures", 5))
        win = "Calib RGB — drag arm in MuJoCo; SPACE=capture, ESC=finish"
        cv2.namedWindow(win)
        print(
            f"[Calib] Pose the arm under gravity compensation so it is visible "
            f"in this RGB view. SPACE to capture ({n_want} needed), ESC to finish early. "
            f"Ctrl+C in the terminal also exits."
        )

        views: List[dict] = []
        while len(views) < n_want:
            self._poll_sensors()
            rgb = self._last_rgb
            q = self._last_q
            if rgb is not None:
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                cv2.putText(
                    vis,
                    f"captured {len(views)}/{n_want}  |  SPACE=capture  ESC=done",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                if q is None:
                    cv2.putText(
                        vis,
                        "waiting for joint_meas...",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 128, 255),
                        2,
                    )
                else:
                    q_arm = np.rad2deg(q[: self.num_arm])
                    q_txt = "arm_deg=[" + ",".join(f"{v:.0f}" for v in q_arm) + "]"
                    cv2.putText(
                        vis,
                        f"{q_txt}  updates={self._joint_update_count}",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 128),
                        1,
                    )
                cv2.imshow(win, vis)
            key = self._wait_key(15)
            if key == 27:  # ESC
                break
            if key != ord(" "):
                continue
            # Re-poll right before freezing so q matches the displayed RGB.
            self._poll_sensors()
            rgb = self._last_rgb
            q = self._last_q
            if rgb is None or q is None:
                print("[Calib] No RGB/q yet — wait a moment and try SPACE again.")
                continue

            # Freeze this frame for SAM2 (keep polling so LCM does not stall).
            snap_rgb = rgb.copy()
            snap_q = q.copy()
            print(
                f"[Calib] Capture {len(views) + 1}/{n_want}  "
                f"arm_q(rad)={np.array2string(snap_q[: self.num_arm], precision=3)}"
            )
            mask = self._sam2_mask(snap_rgb)
            gray = _rgb_to_gray01(snap_rgb)
            I_obs = gray * mask.astype(np.float32)
            mesh = self._fk.mesh_at_q(snap_q)
            views.append(
                {
                    "rgb": snap_rgb,
                    "q": snap_q,
                    "mask": mask,
                    "I_obs": I_obs,
                    "vertices": np.asarray(mesh.vertices, dtype=np.float32).copy(),
                    "faces": np.asarray(mesh.faces, dtype=np.int32).copy(),
                }
            )
            prev = (np.stack([I_obs, I_obs, I_obs], axis=-1) * 255).astype(np.uint8)
            cv2.imshow(
                "Observed gray (masked)", cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)
            )
            self._wait_key(200)
            print(
                f"[Calib] Stored view {len(views)}/{n_want}. "
                "Pose again and SPACE for the next capture."
            )

        cv2.destroyWindow(win)
        if len(views) < 2:
            raise RuntimeError(
                f"Need at least 2 captures for calibration, got {len(views)}"
            )
        print(f"[Calib] Using {len(views)} captured views.")
        return views

    def optimize(self, views: List[dict]) -> np.ndarray:
        assert self._K is not None and self._T_gt_c2b is not None
        for i, v in enumerate(views):
            print(
                f"[Calib] view{i} arm_q={np.array2string(v['q'][: self.num_arm], precision=3)} "
                f"mesh_centroid={v['vertices'].mean(axis=0)}"
            )
        rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        rot_noise = float(self.cfg.get("init_rot_noise_deg", 5.0))
        trans_noise = float(self.cfg.get("init_trans_noise_m", 0.03))
        # Optimize OpenCV cam_in_base (eyeball / EasyHeC), not world2cam.
        T_init = _perturb_cam_in_base(self._T_gt_c2b, rot_noise, trans_noise, rng)
        omega, t = wt_from_se3(T_init)
        omega = omega.cuda().detach().requires_grad_(True)
        t = t.cuda().detach().requires_grad_(True)

        gray = float(self.cfg.get("render_gray", 0.7))
        self._renderer = DiffSilhouetteRenderer(
            self._K, self._H, self._W, gray=gray, device="cuda"
        )
        # EasyHeC / ReactiveGrasp refine_cam_pose_easyhec defaults.
        lr = float(self.cfg.get("lr", 3e-3))
        n_iters = int(self.cfg.get("n_iters", 2000))
        early_stop = int(self.cfg.get("early_stop", 200))
        viz_every = int(self.cfg.get("viz_every", 5))
        opt = torch.optim.Adam([omega, t], lr=lr)

        win = "Calib: obs | rend | |diff|"
        print(
            f"[Calib] Optimizing OpenCV T_cam2base for up to {n_iters} iters "
            f"(lr={lr}, early_stop={early_stop}, "
            f"init noise ~{rot_noise} deg / {trans_noise} m)..."
        )
        best_loss = float("inf")
        best_state = (omega.detach().clone(), t.detach().clone())
        last_improve = 0
        for it in range(n_iters):
            self._check_stop()
            opt.zero_grad()
            T_c2b = se3_from_wt(omega, t)
            # EasyHeC: sum_pixels((rend-mask)^2) / N_views  (not mean-over-pixels).
            loss = torch.zeros((), device="cuda")
            mosaics = []
            for v in views:
                self._renderer.set_mesh(v["vertices"], v["faces"])
                I_rend = self._renderer.render(T_c2b)
                I_obs = torch.as_tensor(
                    v["I_obs"], dtype=torch.float32, device="cuda"
                )
                rend_sil = (I_rend / max(gray, 1e-3)).clamp(0.0, 1.0)
                obs_sil = (I_obs > 1e-3).float()
                loss = loss + torch.sum((obs_sil - rend_sil) ** 2)
                if it % viz_every == 0:
                    o = obs_sil.detach().cpu().numpy()
                    r = rend_sil.detach().cpu().numpy()
                    d = np.abs(o - r)
                    mosaics.append(np.concatenate([o, r, d], axis=1))
            loss = loss / len(views)
            loss.backward()
            opt.step()

            loss_f = float(loss.detach())
            if loss_f + 1e-9 < best_loss:
                best_loss = loss_f
                best_state = (omega.detach().clone(), t.detach().clone())
                last_improve = it

            if it % viz_every == 0:
                panel = np.concatenate(mosaics[: min(3, len(mosaics))], axis=0)
                panel_u8 = (np.clip(panel, 0, 1) * 255).astype(np.uint8)
                panel_bgr = cv2.cvtColor(panel_u8, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    panel_bgr,
                    f"iter {it}  loss={loss_f:.1f}  best={best_loss:.1f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(win, panel_bgr)
                key = self._wait_key(1)
                if key in (ord("q"), 27):
                    self._stop = True
                    self._check_stop()
                print(
                    f"[Calib] iter {it:4d}  loss={loss_f:.3f}  best={best_loss:.3f}"
                )

            if it - last_improve >= early_stop:
                print(
                    f"[Calib] Early stop at iter {it} "
                    f"(no improve for {early_stop} steps)."
                )
                break

        omega.data.copy_(best_state[0])
        t.data.copy_(best_state[1])
        T_cam2base = se3_from_wt(omega, t).detach().cpu().numpy()
        T_world2cam = invSE3(T_cam2base)
        t_err, r_err = _pose_error(T_cam2base, self._T_gt_c2b)
        print("[Calib] Done.")
        print(f"[Calib] GT  T_cam2base (cam_in_base):\n{self._T_gt_c2b}")
        print(f"[Calib] Est T_cam2base (cam_in_base):\n{T_cam2base}")
        print(f"[Calib] Est T_world2cam:\n{T_world2cam}")
        print(f"[Calib] Error vs GT: trans={t_err*1000:.1f} mm, rot={r_err:.2f} deg")
        print("[Calib] Press any key (or Ctrl+C) to close.")
        self._wait_key_until(0)
        cv2.destroyAllWindows()
        return T_cam2base

    def run(self) -> np.ndarray:
        self._install_signal_handlers()
        try:
            views = self.capture_views()
            return self.optimize(views)
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self._restore_signal_handlers()
