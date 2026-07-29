"""Minimum-jerk point-to-point controller (sample → B-spline plan → PD track).

Keyboard (controller terminal)
------------------------------
``1``  sample a random collision-free joint goal (env + self-collision)
``2``  plan with RRT-Connect, B-spline smooth, min-jerk time scale
``3``  PD-track the planned trajectory
``g``  gravity-comp mode (kp=kd=0; float with τ=g(q))
``h``  cancel tracking / hold current pose (PD)
``p`` / ``q``  pause / quit

Viser sliders publish ``p2p_gains_cmd`` (kp/kd/friction scales); controller
echoes live ``p2p_status`` for q/q_des plots and gain readouts.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import fcl
import numpy as np
import pinocchio as pin
from omegaconf import DictConfig
from scipy.interpolate import make_interp_spline

from controller.BaseController import BaseController
from data_type.basic_types.ColInfoData import ColInfoData
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData
from data_type.basic_types.JointTrajData import JointTrajData
from data_type.basic_types.P2PGainsCmdData import P2PGainsCmdData
from data_type.basic_types.P2PStatusData import (
    MODE_GRAV,
    MODE_HOLD,
    MODE_TRACK,
    P2PStatusData,
)
from utils.fcl.getter import get_fcl_geom
from utils.pinocchio.getter import get_fk_link_pose
from utils.planning.path2traj import path_to_min_jerk_trajectory
from utils.planning.rrt_connect import rrt_connect


@dataclass
class PlannedTrajectory:
    """Discrete min-jerk trajectory ready for PD tracking."""

    t: np.ndarray  # (M,)
    q: np.ndarray  # (M, n)
    qd: np.ndarray  # (M, n)
    waypoints: np.ndarray  # (K, n) geometric waypoints used for the B-spline
    duration: float


class MinimumJerkP2PController(BaseController):
    """Hold → sample free goal → B-spline / min-jerk plan → PD track."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._dict_joints = config["dict_joints"]
        self._num_joints = int(config["num_joints"])
        self._n = self._num_joints

        self._robot_col_info_channel = config["sub_manager"]["robot_col_info_channel"]
        self._static_col_info_channel = config["sub_manager"]["static_col_info_channel"]
        self._gains_cmd_channel = config["sub_manager"].get("gains_cmd_channel", "")
        self._target_joint_channel = config["pub_manager"]["target_joint_channel"]
        self._traj_channel = config["pub_manager"]["traj_channel"]
        self._status_channel = config["pub_manager"].get(
            "status_channel", "sw_p2p_status"
        )
        self._traj_publish_stride = int(config.get("traj_publish_stride", 5))
        self._status_publish_stride = int(config.get("status_publish_stride", 20))
        self._status_tick = 0

        self._num_arm = int(config.get("num_arm_joints", 7))
        self._kp_base = np.asarray(config["kp"], dtype=np.float64)
        self._kd_base = np.asarray(config["kd"], dtype=np.float64)
        self._kp_scale = float(config.get("kp_scale", 1.0))
        self._kd_scale = float(config.get("kd_scale", 1.0))
        self._joint_velocity_limit = np.asarray(
            config["joint_velocity_limit"], dtype=np.float64
        )

        # Gravity / friction feedforward (IKQP-style Coulomb).
        self._do_grav_comp = bool(config.get("do_grav_comp", True))
        self._do_friction_comp = bool(config.get("do_friction_comp", True))
        self._friction_phi = float(config.get("friction_phi", 0.03))
        self._Fjc_base = np.asarray(
            config.get("Fjc", np.zeros(self._n)), dtype=np.float64
        )
        if self._Fjc_base.size != self._n:
            raise ValueError(
                f"Fjc length {self._Fjc_base.size} != num_joints {self._n}"
            )
        self._fjc_scale = float(config.get("fjc_scale", 1.0))
        self._arm_damping = float(config.get("arm_damping", 0.1))
        self._ctrl_mode = "hold"  # hold | track | grav
        self._gains_lock = threading.Lock()

        self._safety_dist_thr = float(config.get("safty_dist_thr", 0.02))
        self._col_link_names = list(config["col_link_names"])
        self._obs_names = list(config["obs_names"])
        self._col_pairs = [tuple(p) for p in config["col_pairs"]]
        self._self_col_pairs = [tuple(p) for p in config.get("self_col_pairs", [])]

        self._sample_joint_idx = np.asarray(
            config.get("sample_joint_idx", list(range(7))), dtype=int
        )
        self._joint_limit_margin = float(config.get("joint_limit_margin", 0.05))
        self._sample_max_tries = int(config.get("sample_max_tries", 500))

        self._seg_check_steps = int(config.get("seg_check_steps", 25))
        self._bspline_samples = int(config.get("bspline_samples", 80))
        self._traj_dt = float(config.get("traj_dt", 0.01))
        self._traj_duration = config.get("traj_duration", None)
        if self._traj_duration is not None:
            self._traj_duration = float(self._traj_duration)
        self._traj_duration_scale = float(config.get("traj_duration_scale", 1.25))

        # RRT-Connect
        self._rrt_step_size = float(config.get("rrt_step_size", 0.25))
        self._rrt_max_iters = int(config.get("rrt_max_iters", 2500))
        self._rrt_goal_bias = float(config.get("rrt_goal_bias", 0.05))
        self._rrt_shortcut_iters = int(config.get("rrt_shortcut_iters", 100))

        pin_robot = pin.RobotWrapper.BuildFromURDF(
            config["urdf_path"], list(config["mesh_path"].values())
        )
        self._pin_model = pin_robot.model
        # Separate Data for control vs sampling/planning (Pinocchio Data is not
        # thread-safe; planning runs on a background thread).
        self._pin_data_ctrl = pin_robot.data
        self._pin_data_plan = self._pin_model.createData()
        self._q_lower = self._pin_model.lowerPositionLimit.copy()
        self._q_upper = self._pin_model.upperPositionLimit.copy()

        self._q_default = np.asarray(
            config["default_q_values"], dtype=np.float64
        ).copy()
        self._q = self._q_default.copy()
        self._qd = np.zeros(self._n, dtype=np.float64)
        self._q_hold = self._q_default.copy()

        self._q_goal: Optional[np.ndarray] = None
        self._traj: Optional[PlannedTrajectory] = None
        self._active_traj: Optional[PlannedTrajectory] = None
        self._tracking = False
        self._track_arming = False  # key "3" → start on next control tick
        self._t_traj_start = 0.0
        self._planning = False
        self._hold_seeded = False

        self._meas_lock = threading.Lock()
        self._plan_lock = threading.Lock()
        self._plan_thread: Optional[threading.Thread] = None

        self._fcl_robot: dict = {}
        self._fcl_static: dict = {}

        if type(self) is MinimumJerkP2PController:
            print(
                "[MinJerkP2P] Keys: 1=sample free goal, 2=RRT-Connect+B-spline/min-jerk, "
                "3=PD track, g=grav-comp, h=hold, p=pause, q=quit"
            )

    # ------------------------------------------------------------------ init
    def initialize(self) -> None:
        self._fcl_robot = self._wait_for_col_geoms(
            self.intr_sub_que_dict[self._robot_col_info_channel], "robot"
        )
        self._fcl_static = self._wait_for_col_geoms(
            self.extr_sub_que_dict[self._static_col_info_channel], "static"
        )
        # Seed hold pose from default; first measurement will refresh it.
        self._q_hold = self._q_default.copy()
        print("[MinJerkP2P] Initialized with FCL geoms.")

    def _wait_for_col_geoms(self, que, label: str, timeout_s: float = 2.0) -> dict:
        tic = time.time()
        while True:
            if not que.empty():
                msg = que.get()
                if not isinstance(msg, ColInfoData):
                    raise ValueError(
                        f"Expected ColInfoData for {label}, got {type(msg)}"
                    )
                _, dict_col = msg.get_data()
                return {
                    name: {
                        "fcl_geom": get_fcl_geom(**g),
                        "offset": np.asarray(g["offset"], dtype=float).reshape(4, 4),
                    }
                    for name, g in dict_col.items()
                }
            if time.time() - tic > timeout_s:
                raise TimeoutError(f"Timeout waiting for {label} ColInfo.")
            time.sleep(0.001)

    # -------------------------------------------------------------- keyboard
    def _handle_key(self, key: str) -> None:
        if key in ("p", "q"):
            super()._handle_key(key)
            return

        if key == "h":
            self._enter_hold_mode(hold_current=True)
            return

        if key == "g":
            self._enter_grav_mode()
            return

        if key == "1":
            self._action_sample_goal()
            return
        if key == "2":
            self._action_plan()
            return
        if key == "3":
            self._action_track()
            return

        print(f"[MinJerkP2P] Unused key: {key!r}")

    def _enter_hold_mode(self, hold_current: bool = True) -> None:
        self._cancel_tracking(hold_current=hold_current)
        self._ctrl_mode = "hold"
        print("[MinJerkP2P] Mode → hold (PD).")

    def _enter_grav_mode(self) -> None:
        self._cancel_tracking(hold_current=True)
        self._ctrl_mode = "grav"
        print("[MinJerkP2P] Mode → grav (kp=kd=0, τ=g(q)).")

    def _action_sample_goal(self) -> None:
        with self._plan_lock:
            if self._planning:
                print("[MinJerkP2P] Busy planning/sampling — wait.")
                return
            self._planning = True
            self._tracking = False
            self._track_arming = False
            self._active_traj = None

        with self._meas_lock:
            q_seed = self._q.copy()

        def _worker():
            try:
                q_goal = self._sample_collision_free_q(q_seed)
                with self._plan_lock:
                    if q_goal is None:
                        print(
                            f"[MinJerkP2P] Failed to sample a free config "
                            f"in {self._sample_max_tries} tries."
                        )
                    else:
                        self._q_goal = q_goal
                        self._traj = None
                        print(
                            "[MinJerkP2P] Sampled collision-free goal "
                            f"(arm={np.array2string(q_goal[:7], precision=3, separator=', ')}). "
                            "Press 2 to plan."
                        )
                        self._publish_target_joint(q_goal)
            finally:
                with self._plan_lock:
                    self._planning = False

        self._plan_thread = threading.Thread(
            target=_worker, name="min_jerk_sample", daemon=True
        )
        self._plan_thread.start()
        print("[MinJerkP2P] Sampling collision-free goal…")

    def _action_plan(self) -> None:
        with self._plan_lock:
            if self._q_goal is None:
                print("[MinJerkP2P] No goal — press 1 to sample first.")
                return
            if self._planning:
                print("[MinJerkP2P] Already planning.")
                return
            self._planning = True
            self._tracking = False
            self._track_arming = False
            self._active_traj = None
            q_goal = self._q_goal.copy()

        with self._meas_lock:
            q_start = self._q.copy()
        with self._plan_lock:
            # Keep hold synced to the planned start so key "3" has no jump.
            self._q_hold = q_start.copy()

        def _worker():
            try:
                traj = self._plan_bspline_min_jerk(q_start, q_goal)
                with self._plan_lock:
                    self._traj = traj
                if traj is not None:
                    print(
                        f"[MinJerkP2P] Plan OK: {traj.waypoints.shape[0]} waypoints, "
                        f"T={traj.duration:.2f}s, {traj.t.size} samples. Press 3 to track."
                    )
                    self._publish_joint_traj(traj)
                else:
                    print("[MinJerkP2P] Planning failed (no collision-free path).")
            finally:
                with self._plan_lock:
                    self._planning = False

        self._plan_thread = threading.Thread(
            target=_worker, name="min_jerk_plan", daemon=True
        )
        self._plan_thread.start()
        print("[MinJerkP2P] Planning in background…")

    def _action_track(self) -> None:
        """Arm tracking; actual start timestamp is taken in ``_update``."""
        with self._plan_lock:
            if self._planning:
                print("[MinJerkP2P] Still planning — wait.")
                return
            if self._traj is None:
                print("[MinJerkP2P] No plan — press 2 to plan first.")
                return
            if self._tracking or self._track_arming:
                print("[MinJerkP2P] Already tracking / arming.")
                return
            # Do NOT stamp time here — keyboard runs before _update_time(), so
            # t_local would jump on the first control tick (looks like a twitch).
            self._track_arming = True
            self._ctrl_mode = "track"
            T = self._traj.duration
        print(f"[MinJerkP2P] Tracking armed (T={T:.2f}s) — starts next control tick.")

    def _cancel_tracking(self, hold_current: bool = True) -> None:
        with self._meas_lock:
            q = self._q.copy()
        with self._plan_lock:
            self._tracking = False
            self._track_arming = False
            self._active_traj = None
            if hold_current:
                self._q_hold = q
            if self._ctrl_mode == "track":
                self._ctrl_mode = "hold"

    def _publish_target_joint(
        self, q: np.ndarray, *, float_hand: bool = False
    ) -> None:
        """Publish sampled goal as joint_meas_t (q filled; qd/tau zero).

        ``float_hand=True`` marks a palm+fingers-only preview (tau[0]=1) so the
        visualizer can float the hand at ``T_des`` without showing the arm.
        """
        if self.pub_que_dict is None:
            return
        que = self.pub_que_dict.get(self._target_joint_channel)
        if que is None:
            return
        tau = np.zeros(self._n, dtype=np.float64)
        if float_hand:
            tau[0] = 1.0
        msg = JointMeasData(num_joints=self._n, name=self._target_joint_channel)
        msg.set_data(
            float(self._cur_time),
            q,
            np.zeros(self._n, dtype=np.float64),
            tau,
        )
        try:
            while not que.empty():
                que.get_nowait()
        except Exception:
            pass
        try:
            que.put_nowait(msg)
        except Exception:
            pass

    def _publish_joint_traj(self, traj: PlannedTrajectory) -> None:
        """Publish planned (t, q, qd), optionally downsampled for the wire."""
        if self.pub_que_dict is None:
            return
        que = self.pub_que_dict.get(self._traj_channel)
        if que is None:
            return
        stride = max(1, self._traj_publish_stride)
        idx = np.arange(0, traj.t.size, stride)
        if idx[-1] != traj.t.size - 1:
            idx = np.append(idx, traj.t.size - 1)
        data = JointTrajData(name=self._traj_channel)
        data.set_data(
            float(self._cur_time),
            traj.t[idx],
            traj.q[idx],
            traj.qd[idx],
        )
        try:
            while not que.empty():
                que.get_nowait()
        except Exception:
            pass
        try:
            que.put_nowait(data)
        except Exception:
            pass

    # ------------------------------------------------------------- measurements
    def _check_and_get_data_from_que(self) -> None:
        for joint_cfg in self._dict_joints.values():
            channel = joint_cfg["joint_meas_channel"]
            idx = joint_cfg["list_joint_idx"]
            if self.intr_sub_que_dict[channel].empty():
                continue
            meas = self.intr_sub_que_dict[channel].get()
            if not isinstance(meas, JointMeasData):
                raise ValueError(f"Expected JointMeasData on '{channel}'")
            _, q, qd, _ = meas.get_data()
            with self._meas_lock:
                self._q[idx] = q
                self._qd[idx] = qd

    # --------------------------------------------------------- collision check
    def _robot_fcl_objects(self, q: np.ndarray) -> dict:
        out = {}
        for name in self._col_link_names:
            T = get_fk_link_pose(q, self._pin_model, self._pin_data_plan, name)
            entry = self._fcl_robot[name]
            tf = T @ entry["offset"]
            out[name] = fcl.CollisionObject(
                entry["fcl_geom"],
                fcl.Transform(tf[:3, :3], tf[:3, 3]),
            )
        return out

    def _static_fcl_objects(self) -> dict:
        out = {}
        for name in self._obs_names:
            entry = self._fcl_static[name]
            T = entry["offset"]
            out[name] = fcl.CollisionObject(
                entry["fcl_geom"],
                fcl.Transform(T[:3, :3], T[:3, 3]),
            )
        return out

    @staticmethod
    def _fcl_distance(a: fcl.CollisionObject, b: fcl.CollisionObject) -> float:
        req = fcl.DistanceRequest(enable_nearest_points=False)
        res = fcl.DistanceResult()
        fcl.distance(a, b, req, res)
        return float(res.min_distance)

    def _min_clearance(self, q: np.ndarray) -> float:
        """Minimum FCL distance over env pairs and self-collision pairs."""
        robot = self._robot_fcl_objects(q)
        static = self._static_fcl_objects()
        d_min = np.inf
        for link, obs in self._col_pairs:
            d_min = min(d_min, self._fcl_distance(robot[link], static[obs]))
        for a, b in self._self_col_pairs:
            d_min = min(d_min, self._fcl_distance(robot[a], robot[b]))
        return float(d_min)

    def _is_collision_free(self, q: np.ndarray) -> bool:
        return self._min_clearance(q) >= self._safety_dist_thr

    def _sample_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = self._q_lower.copy()
        hi = self._q_upper.copy()
        m = self._joint_limit_margin
        lo[self._sample_joint_idx] = lo[self._sample_joint_idx] + m
        hi[self._sample_joint_idx] = hi[self._sample_joint_idx] - m
        # Guard against inverted bounds on short joints.
        swap = lo > hi
        mid = 0.5 * (self._q_lower + self._q_upper)
        lo[swap] = mid[swap]
        hi[swap] = mid[swap]
        return lo, hi

    def _sample_collision_free_q(self, q_template: np.ndarray) -> Optional[np.ndarray]:
        """Uniform sample on ``sample_joint_idx``; other joints stay at template."""
        lo, hi = self._sample_bounds()
        idx = self._sample_joint_idx
        for _ in range(self._sample_max_tries):
            q = q_template.copy()
            q[idx] = np.random.uniform(lo[idx], hi[idx])
            if self._is_collision_free(q):
                return q
        return None

    def _segment_collision_free(self, qa: np.ndarray, qb: np.ndarray) -> bool:
        n = max(2, self._seg_check_steps)
        for a in np.linspace(0.0, 1.0, n):
            if not self._is_collision_free((1.0 - a) * qa + a * qb):
                return False
        return True

    # --------------------------------------------------------------- planning
    def _find_waypoints(self, q0: np.ndarray, qg: np.ndarray) -> Optional[np.ndarray]:
        """Collision-free polyline via RRT-Connect (+ shortcutting)."""
        path = rrt_connect(
            q0,
            qg,
            is_free=self._is_collision_free,
            segment_free=self._segment_collision_free,
            sample_free=lambda: self._sample_collision_free_q(q0),
            step_size=self._rrt_step_size,
            max_iters=self._rrt_max_iters,
            goal_bias=self._rrt_goal_bias,
            shortcut_iters=self._rrt_shortcut_iters,
        )
        if path is None:
            print(f"[MinJerkP2P] RRT-Connect failed after {self._rrt_max_iters} iters.")
        else:
            print(f"[MinJerkP2P] RRT-Connect OK: {path.shape[0]} waypoints.")
        return path

    def _bspline_path(self, waypoints: np.ndarray, n_samples: int) -> np.ndarray:
        """Cubic (or lower) interpolating B-spline through waypoints."""
        wp = np.asarray(waypoints, dtype=np.float64)
        K, n = wp.shape
        if K == 2:
            a = np.linspace(0.0, 1.0, n_samples)[:, None]
            return (1.0 - a) * wp[0] + a * wp[1]

        # Chord-length parameterization.
        ds = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(ds)])
        if u[-1] < 1e-12:
            return np.repeat(wp[:1], n_samples, axis=0)
        u = u / u[-1]
        # Drop duplicate consecutive u (degenerate segments).
        keep = np.ones(K, dtype=bool)
        keep[1:] = np.diff(u) > 1e-12
        u = u[keep]
        wp = wp[keep]
        k = int(min(3, len(u) - 1))
        spl = make_interp_spline(u, wp, k=k)
        uu = np.linspace(0.0, 1.0, n_samples)
        return np.asarray(spl(uu), dtype=np.float64)

    def _path_is_collision_free(self, q_path: np.ndarray) -> bool:
        return all(self._is_collision_free(q) for q in q_path)

    def _choose_duration(self, q_path: np.ndarray) -> float:
        if self._traj_duration is not None and self._traj_duration > 0:
            return self._traj_duration
        # Conservative rest-to-rest min-jerk duration from peak |Δq|/vlim.
        dq = np.max(np.abs(q_path[-1] - q_path[0]))
        # Also account for path length vs straight-line.
        ds = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
        L = float(np.sum(ds))
        straight = float(np.linalg.norm(q_path[-1] - q_path[0]))
        stretch = L / max(straight, 1e-6)

        vlim = self._joint_velocity_limit
        T_joint = (15.0 / 8.0) * np.abs(q_path[-1] - q_path[0]) / np.maximum(vlim, 1e-6)
        T = float(np.max(T_joint)) * stretch * self._traj_duration_scale
        return max(T, 0.5)

    def _plan_bspline_min_jerk(
        self, q0: np.ndarray, qg: np.ndarray
    ) -> Optional[PlannedTrajectory]:
        waypoints = self._find_waypoints(q0, qg)
        if waypoints is None:
            return None

        q_path = self._bspline_path(waypoints, self._bspline_samples)
        if not self._path_is_collision_free(q_path):
            # B-spline may cut corners — fall back to polyline densification.
            print("[MinJerkP2P] B-spline collided; densifying polyline instead.")
            pieces = []
            for i in range(len(waypoints) - 1):
                n = max(2, self._bspline_samples // max(1, len(waypoints) - 1))
                a = np.linspace(0.0, 1.0, n, endpoint=(i == len(waypoints) - 2))[
                    :, None
                ]
                pieces.append((1.0 - a) * waypoints[i] + a * waypoints[i + 1])
            q_path = np.vstack(pieces)
            if not self._path_is_collision_free(q_path):
                return None

        T = self._choose_duration(q_path)
        t, q, qd, _ = path_to_min_jerk_trajectory(q_path, T=T, dt=self._traj_dt)
        return PlannedTrajectory(
            t=t, q=q, qd=qd, waypoints=waypoints, duration=float(t[-1])
        )

    def _eval_traj(self, t_local: float, traj: PlannedTrajectory):
        """Linear interpolate discrete (q, qd) tables."""
        t = traj.t
        if t_local <= t[0]:
            return traj.q[0].copy(), traj.qd[0].copy(), False
        if t_local >= t[-1]:
            return traj.q[-1].copy(), np.zeros(self._n), True
        i = int(np.searchsorted(t, t_local, side="right") - 1)
        i = max(0, min(i, len(t) - 2))
        a = (t_local - t[i]) / max(t[i + 1] - t[i], 1e-12)
        q = (1.0 - a) * traj.q[i] + a * traj.q[i + 1]
        qd = (1.0 - a) * traj.qd[i] + a * traj.qd[i + 1]
        return q, qd, False

    # --------------------------------------------------------------- control
    def _begin_tracking_if_armed(self, q: np.ndarray) -> None:
        """Start trajectory on the control thread after track key armed it.

        Stamping ``_t_traj_start`` here (after ``_update_time``) avoids a large
        ``t_local`` on the first tick when the keyboard handler ran on a stale
        clock. Active traj tables are copied so a later replan cannot mutate
        the tables the tracker is reading.
        """
        with self._plan_lock:
            if not self._track_arming or self._traj is None:
                return
            traj = self._traj
            self._track_arming = False
            self._tracking = True
            self._t_traj_start = float(self._cur_time)
            self._active_traj = PlannedTrajectory(
                t=np.asarray(traj.t, dtype=np.float64).copy(),
                q=np.asarray(traj.q, dtype=np.float64).copy(),
                qd=np.asarray(traj.qd, dtype=np.float64).copy(),
                waypoints=np.asarray(traj.waypoints, dtype=np.float64).copy(),
                duration=float(traj.duration),
            )
            q0 = self._active_traj.q[0]
            T = self._active_traj.duration

        err = float(np.linalg.norm(q - q0))
        if err > 0.15:
            print(
                f"[MinJerkP2P] Warning: start error ||q-q0||={err:.3f} rad "
                f"(pose drifted since plan). Tracking from traj[0] anyway."
            )
        print(
            f"[MinJerkP2P] Tracking started at t={self._t_traj_start:.3f}s (T={T:.2f}s)."
        )

    def _effective_gains(self):
        with self._gains_lock:
            kp = self._kp_base * self._kp_scale
            kd = self._kd_base * self._kd_scale
            fjc = self._Fjc_base * self._fjc_scale
            phi = self._friction_phi
            do_fric = self._do_friction_comp
            do_grav = self._do_grav_comp
            kp_s = self._kp_scale
            kd_s = self._kd_scale
            fjc_s = self._fjc_scale
        return kp, kd, fjc, phi, do_fric, do_grav, kp_s, kd_s, fjc_s

    def _friction_compensation(self, qd_ref: np.ndarray, fjc: np.ndarray, phi: float):
        phi = max(float(phi), 1e-6)
        return fjc * np.clip(qd_ref / phi, -1.0, 1.0)

    def _poll_gains_cmd(self) -> None:
        ch = self._gains_cmd_channel
        if not ch or ch not in self.intr_sub_que_dict:
            return
        que = self.intr_sub_que_dict[ch]
        cmd = None
        while not que.empty():
            cmd = que.get()
        if not isinstance(cmd, P2PGainsCmdData):
            return
        with self._gains_lock:
            self._kp_scale = float(cmd.kp_scale)
            self._kd_scale = float(cmd.kd_scale)
            self._fjc_scale = float(cmd.fjc_scale)
            self._friction_phi = float(cmd.friction_phi)
            self._do_friction_comp = bool(cmd.do_friction_comp)

    def _publish_status(
        self,
        mode: int,
        q,
        q_des,
        qd,
        qd_des,
        kp,
        kd,
        fjc,
        *,
        phi: float,
        kp_scale: float,
        kd_scale: float,
        fjc_scale: float,
        do_fric: bool,
        do_grav: bool,
    ) -> None:
        if self.pub_que_dict is None:
            return
        que = self.pub_que_dict.get(self._status_channel)
        if que is None:
            return
        self._status_tick += 1
        if self._status_tick % max(1, self._status_publish_stride) != 0:
            return
        status = P2PStatusData(num_joints=self._n, name=self._status_channel)
        status.set_data(
            float(self._cur_time),
            mode,
            q,
            q_des,
            qd,
            qd_des,
            kp,
            kd,
            fjc,
            num_arm=self._num_arm,
            friction_phi=phi,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            fjc_scale=fjc_scale,
            do_friction_comp=do_fric,
            do_grav_comp=do_grav,
        )
        try:
            while not que.empty():
                que.get_nowait()
        except Exception:
            pass
        try:
            que.put_nowait(status)
        except Exception:
            pass

    def _update(self) -> None:
        self._poll_gains_cmd()

        with self._meas_lock:
            q = self._q.copy()
            qd = self._qd.copy()

        with self._plan_lock:
            if not self._hold_seeded:
                self._q_hold = q.copy()
                self._hold_seeded = True

        self._begin_tracking_if_armed(q)

        kp, kd, fjc, phi, do_fric, do_grav, kp_s, kd_s, fjc_s = self._effective_gains()

        # Gravity-comp float: no PD, τ = g(q) - b qd_arm (+ friction on measured qd).
        if self._ctrl_mode == "grav":
            q_des = q.copy()
            qd_des = np.zeros(self._n, dtype=np.float64)
            kp_cmd = np.zeros(self._n, dtype=np.float64)
            kd_cmd = np.zeros(self._n, dtype=np.float64)
            if do_grav:
                tau_ff = pin.computeGeneralizedGravity(
                    self._pin_model, self._pin_data_ctrl, q
                )
            else:
                tau_ff = np.zeros(self._n, dtype=np.float64)
            arm = slice(0, self._num_arm)
            tau_ff[arm] = tau_ff[arm] - self._arm_damping * qd[arm]
            if do_fric:
                tau_ff = tau_ff + self._friction_compensation(qd, fjc, phi)
            mode_id = MODE_GRAV
        else:
            with self._plan_lock:
                tracking = self._tracking
                traj = (
                    self._active_traj if self._active_traj is not None else self._traj
                )
                t0 = self._t_traj_start

            if tracking and traj is not None:
                q_des, qd_des, done = self._eval_traj(float(self._cur_time) - t0, traj)
                if done:
                    with self._plan_lock:
                        self._tracking = False
                        self._active_traj = None
                        self._q_hold = q_des.copy()
                        self._ctrl_mode = "hold"
                    print("[MinJerkP2P] Trajectory finished — holding goal.")
                    tracking = False
                mode_id = MODE_TRACK if tracking else MODE_HOLD
            else:
                with self._plan_lock:
                    q_des = self._q_hold.copy()
                qd_des = np.zeros(self._n, dtype=np.float64)
                mode_id = MODE_HOLD

            kp_cmd = kp
            kd_cmd = kd
            if do_grav:
                tau_ff = pin.computeGeneralizedGravity(
                    self._pin_model, self._pin_data_ctrl, q
                )
            else:
                tau_ff = np.zeros(self._n, dtype=np.float64)
            if do_fric:
                tau_ff = tau_ff + self._friction_compensation(qd_des, fjc, phi)

        cmd = JointCtrlData(num_joints=self._n)
        cmd.set_data(self._cur_time, q_des, qd_des, tau_ff, kp_cmd, kd_cmd)
        self.ctrl_pub_que.put(cmd)

        self._publish_status(
            mode_id,
            q,
            q_des,
            qd,
            qd_des,
            kp_cmd,
            kd_cmd,
            fjc,
            phi=phi,
            kp_scale=kp_s,
            kd_scale=kd_s,
            fjc_scale=fjc_s,
            do_fric=do_fric,
            do_grav=do_grav,
        )
