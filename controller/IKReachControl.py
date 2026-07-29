"""Stepped palm SE(3) + hand sample → IK → joint-space impedance to q*.

Keyboard
--------
``1``  sample palm pose + finger joints (Viser: floating palm+fingers at T_des)
``2``  solve SE(3) IK (Viser: full arm+hand ghost at q*)
``3``  joint-space impedance toward ``q*`` (no path planning)
``g`` / ``h`` / ``p`` / ``q``  as in min-jerk P2P
"""

from __future__ import annotations

import threading

import numpy as np
from omegaconf import DictConfig

from controller.MinimumJerkP2PControl import MinimumJerkP2PController
from data_type.basic_types.SE3PoseData import SE3PoseData
from utils.pinocchio.getter import get_fk_link_pose
from utils.pinocchio.ik_se3 import make_SE3, random_so3, solve_frame_ik_se3


class IKReachController(MinimumJerkP2PController):
    """Task-space sample + SE(3) IK, then joint impedance to the IK solution."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._task_frame = str(config.get("task_frame", "palm"))
        if not self._pin_model.existFrame(self._task_frame):
            raise ValueError(f"task_frame {self._task_frame!r} missing in URDF")
        self._frame_id = self._pin_model.getFrameId(self._task_frame)

        self._ik_tol = float(config.get("ik_tol", 1e-4))
        self._ik_max_iters = int(config.get("ik_max_iters", 200))
        self._ik_damp = float(config.get("ik_damping", 1e-4))
        self._ik_step = float(config.get("ik_step", 0.5))

        self._pose_pos_min = np.asarray(
            config.get("pose_pos_min", [0.30, -0.25, 0.15]), dtype=np.float64
        )
        self._pose_pos_max = np.asarray(
            config.get("pose_pos_max", [0.65, 0.25, 0.55]), dtype=np.float64
        )
        self._pose_rot_noise_rad = float(config.get("pose_rot_noise_rad", 0.5))

        self._hand_joint_idx = np.asarray(
            config.get(
                "hand_sample_joint_idx",
                list(range(self._num_arm, self._n)),
            ),
            dtype=int,
        )
        self._arm_joint_idx = np.arange(self._num_arm, dtype=int)
        self._rng = np.random.default_rng(int(config.get("sample_seed", 0)) or None)

        self._palm_pose_channel = config["pub_manager"].get("palm_pose_channel", "")
        self._T_des: np.ndarray | None = None
        self._q_hand: np.ndarray | None = None
        self._last_ik_err = np.inf
        self._ik_ready = False

        print(
            f"[IKReach] frame={self._task_frame}. "
            "Keys: 1=sample palm+hand, 2=IK, 3=joint impedance to q*, g/h/p/q"
        )

    def _handle_key(self, key: str) -> None:
        if key == "1":
            self._action_sample()
            return
        if key == "2":
            self._action_ik()
            return
        if key == "3":
            self._action_start_impedance()
            return
        if key == "4":
            print("[IKReach] Key 4 unused — press 3 for joint impedance to q*.")
            return
        super()._handle_key(key)

    def _publish_palm_pose(self, T: np.ndarray) -> None:
        if self.pub_que_dict is None or not self._palm_pose_channel:
            return
        que = self.pub_que_dict.get(self._palm_pose_channel)
        if que is None:
            return
        msg = SE3PoseData(name=self._palm_pose_channel)
        msg.set_from_matrix(float(self._cur_time), T)
        try:
            while not que.empty():
                que.get_nowait()
        except Exception:
            pass
        try:
            que.put_nowait(msg)
        except Exception:
            pass

    def _reset_pipeline_locked(self) -> None:
        """Clear IK / impedance-reach state. Caller must hold ``_plan_lock``."""
        self._tracking = False
        self._track_arming = False
        self._active_traj = None
        self._traj = None
        self._q_goal = None
        self._ik_ready = False
        if self._ctrl_mode == "track":
            self._ctrl_mode = "hold"

    # -------------------------------------------------------------- key 1
    def _action_sample(self) -> None:
        """Sample T_des + q_hand; Viser floats palm+fingers at T_des (no arm)."""
        with self._plan_lock:
            if self._planning:
                print("[IKReach] Busy — wait.")
                return
            self._reset_pipeline_locked()

        with self._meas_lock:
            q_now = self._q.copy()

        T_des, q_hand = self._sample_palm_and_hand(q_now)
        q_viz = np.zeros(self._n, dtype=np.float64)
        q_viz[self._hand_joint_idx] = q_hand

        with self._plan_lock:
            self._T_des = T_des
            self._q_hand = q_hand
            self._q_hold = q_now.copy()
            self._ctrl_mode = "hold"

        self._publish_target_joint(q_viz, float_hand=True)
        self._publish_palm_pose(T_des)
        print(
            f"[IKReach] Sampled palm p={np.array2string(T_des[:3, 3], precision=3)} "
            f"+ hand. Press 2 for IK."
        )

    # -------------------------------------------------------------- key 2
    def _action_ik(self) -> None:
        with self._plan_lock:
            if self._T_des is None or self._q_hand is None:
                print("[IKReach] No sample — press 1 first.")
                return
            if self._planning:
                print("[IKReach] Busy — wait.")
                return
            self._planning = True
            self._tracking = False
            self._track_arming = False
            self._active_traj = None
            self._traj = None
            self._q_goal = None
            self._ik_ready = False
            self._ctrl_mode = "hold"
            T_des = self._T_des.copy()
            q_hand = self._q_hand.copy()

        with self._meas_lock:
            q_seed = self._q.copy()

        def _worker():
            try:
                q0 = q_seed.copy()
                q0[self._hand_joint_idx] = q_hand
                q_sol, err, ok = solve_frame_ik_se3(
                    self._pin_model,
                    self._pin_data_plan,
                    q0,
                    T_des,
                    self._frame_id,
                    free_joint_idx=self._arm_joint_idx,
                    max_iters=self._ik_max_iters,
                    tol=self._ik_tol,
                    damp=self._ik_damp,
                    step=self._ik_step,
                    q_lower=self._q_lower,
                    q_upper=self._q_upper,
                )
                self._last_ik_err = err
                if not ok or q_sol is None:
                    print(
                        f"[IKReach] IK failed (||ξ||={err:.2e}). "
                        "Resample with 1 or relax pose box."
                    )
                    return
                if not self._is_collision_free(q_sol):
                    print(
                        f"[IKReach] IK OK (||ξ||={err:.2e}) but config in collision. "
                        "Resample with 1."
                    )
                    return
                with self._plan_lock:
                    self._q_goal = q_sol
                    self._ik_ready = True
                    self._q_hold = q_seed.copy()
                    self._ctrl_mode = "hold"
                self._publish_target_joint(q_sol)
                self._publish_palm_pose(T_des)
                print(
                    f"[IKReach] IK OK (||ξ||={err:.2e}), collision-free. "
                    "Press 3 for joint impedance to q*."
                )
            finally:
                with self._plan_lock:
                    self._planning = False

        self._plan_thread = threading.Thread(
            target=_worker, name="ik_reach_ik", daemon=True
        )
        self._plan_thread.start()
        print("[IKReach] Solving SE(3) IK…")

    # -------------------------------------------------------------- key 3
    def _action_start_impedance(self) -> None:
        """Regulate about ``q*`` with joint-space spring–damper (no path plan).

        Plant law ``τ = kp(q* − q) − kd q̇ + g(q)`` is joint impedance with
        ``K=kp``, ``D=kd`` (same gains as the P2P hold / impedance tutorials).
        """
        with self._plan_lock:
            if self._planning:
                print("[IKReach] Busy — wait.")
                return
            if not self._ik_ready or self._q_goal is None:
                print("[IKReach] No IK goal — press 2 first.")
                return
            self._tracking = False
            self._track_arming = False
            self._active_traj = None
            self._traj = None
            q_star = self._q_goal.copy()
            self._q_hold = q_star
            self._ctrl_mode = "hold"

        with self._meas_lock:
            q_now = self._q.copy()
        err0 = float(np.linalg.norm(q_now - q_star))
        print(
            f"[IKReach] Joint impedance → q*  (||q−q*||={err0:.3f} rad). "
            "Press h to hold current, 1 to resample."
        )

    def _sample_palm_and_hand(self, q_seed: np.ndarray):
        """Random world palm SE(3) and finger joint vector."""
        p = self._rng.uniform(self._pose_pos_min, self._pose_pos_max)
        T_seed = get_fk_link_pose(
            q_seed, self._pin_model, self._pin_data_plan, self._task_frame
        )
        R = T_seed[:3, :3] @ random_so3(self._rng, self._pose_rot_noise_rad)
        T_des = make_SE3(R, p)

        q_hand = np.zeros(self._hand_joint_idx.size, dtype=np.float64)
        margin = self._joint_limit_margin
        for k, j in enumerate(self._hand_joint_idx):
            lo = float(self._q_lower[j] + margin)
            hi = float(self._q_upper[j] - margin)
            if hi < lo:
                lo, hi = float(self._q_lower[j]), float(self._q_upper[j])
            q_hand[k] = float(self._rng.uniform(lo, hi))
        return T_des, q_hand
