"""Collision-aware gravity compensation with a QP repulsive planner."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue

import fcl
import numpy as np
import osqp
import pinocchio as pin
from omegaconf import DictConfig
from scipy import sparse

from controller.BaseController import BaseController
from data_type.basic_types.ColInfoData import ColInfoData
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData
from utils.fcl.getter import get_fcl_geom
from utils.lie.kinematics import get_point_Jacobian
from utils.pinocchio.getter import get_fk_link_poses, get_link_Jacobians


@dataclass
class PlanResult:
    """Latest output from the planning thread."""

    state: str
    qd_des: np.ndarray
    success: bool


class CollisionAwareGravCompController(BaseController):
    """Gravity compensation with FCL distance checks and a repulsive QP.

    Threading
    ---------
    - Control loop (``_update``) publishes torque commands.
    - Planning thread (``_plan_loop``) runs FCL + OSQP at ``plan_freq``.

    Cross-thread communication (no shared plan flags):
    - ``_meas_lock`` protects joint-state snapshots for the planner.
    - ``_plan_result_que`` (maxsize=1) delivers the latest ``PlanResult``.
    - Planner and control each use their own Pinocchio ``Data`` object.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._static_col_info_channel = config["sub_manager"]["static_col_info_channel"]
        self._robot_col_info_channel = config["sub_manager"]["robot_col_info_channel"]

        self._dict_joints = config["dict_joints"]
        self._num_joints = int(config["num_joints"])
        self._n_x = self._num_joints

        self._joint_velocity_limit = np.asarray(
            config["joint_velocity_limit"], dtype=np.float64
        )
        self._kp = np.asarray(config["kp"], dtype=np.float64)
        self._kd = np.asarray(config["kd"], dtype=np.float64)
        self._kp_multiplier = np.asarray(
            config.get(
                "kp_multiplier",
                [1.0] * self._num_joints,
            ),
            dtype=np.float64,
        )

        # Keep original config key spellings.
        self._safety_dist_thr = float(config.get("safty_dist_thr", 0.01))
        self._margin_thr = float(config.get("magin_thr", 0.05))
        self._repulsive_coeff = float(config.get("repulsive_coeff", 10.0))

        self._plan_freq = float(config["plan_freq"])
        self._plan_dt = 1.0 / self._plan_freq
        self._plan_horizon = float(config["plan_horizon"])

        self._col_link_names = list(config["col_link_names"])
        self._obs_names = list(config["obs_names"])
        self._col_pairs = [list(p) for p in config["col_pairs"]]

        pin_robot = pin.RobotWrapper.BuildFromURDF(
            config["urdf_path"], list(config["mesh_path"].values())
        )
        self._pin_model = pin_robot.model
        # Separate Data for each thread — Pinocchio Data is not thread-safe.
        self._pin_data_ctrl = pin_robot.data
        self._pin_data_plan = self._pin_model.createData()
        self._q_lower_limit = self._pin_model.lowerPositionLimit.copy()
        self._q_upper_limit = self._pin_model.upperPositionLimit.copy()

        self._q = np.asarray(config["default_q_values"], dtype=np.float64).copy()
        self._qd = np.zeros(self._num_joints, dtype=np.float64)

        # Held by the control thread; refreshed from the plan-result queue.
        self._qd_des = np.zeros(self._num_joints, dtype=np.float64)
        self._ctrl_state = "grav_comp"
        self._last_logged_state = "grav_comp"

        self._meas_lock = threading.Lock()
        self._plan_result_que: Queue[PlanResult] = Queue(maxsize=1)

        self._stop_event = threading.Event()
        self._plan_thread = threading.Thread(
            target=self._plan_loop, name="col_aware_plan", daemon=False
        )

        self._fcl_static_col_geoms: dict = {}
        self._fcl_robot_col_geoms: dict = {}

    # ------------------------------------------------------------------ init
    def initialize(self) -> None:
        """Block until robot/static ColInfo messages arrive, then build FCL geoms."""
        self._fcl_robot_col_geoms = self._wait_for_col_geoms(
            self.intr_sub_que_dict[self._robot_col_info_channel],
            label="robot",
        )
        self._fcl_static_col_geoms = self._wait_for_col_geoms(
            self.extr_sub_que_dict[self._static_col_info_channel],
            label="static",
        )
        print("[ColAwareGravComp] Initialized with FCL geoms.")

    def _wait_for_col_geoms(
        self, que: Queue, label: str, timeout_s: float = 2.0
    ) -> dict:
        tic = time.time()
        while True:
            if not que.empty():
                msg = que.get()
                if not isinstance(msg, ColInfoData):
                    raise ValueError(
                        f"Expected ColInfoData for {label}, got {type(msg)}"
                    )
                _, dict_col_info = msg.get_data()
                return {
                    name: {
                        "fcl_geom": get_fcl_geom(**col_geom),
                        "offset": col_geom["offset"],
                    }
                    for name, col_geom in dict_col_info.items()
                }
            if time.time() - tic > timeout_s:
                raise TimeoutError(
                    f"Timeout waiting for {label} collision information."
                )
            time.sleep(0.001)

    # ----------------------------------------------------------- thread hooks
    def _sub_thread_start(self) -> None:
        if not self._plan_thread.is_alive():
            self._plan_thread.start()
            print("[ColAwareGravComp] Planning thread started.")

    def _sub_thread_stop(self) -> None:
        if self._plan_thread.is_alive():
            self._stop_event.set()
            self._plan_thread.join(timeout=2.0)
            print("[ColAwareGravComp] Planning thread stopped.")

    # ------------------------------------------------------------- measurements
    def _check_and_get_data_from_que(self) -> None:
        """Pull latest joint measurements into ``_q`` / ``_qd``."""
        for joint_cfg in self._dict_joints.values():
            channel = joint_cfg["joint_meas_channel"]
            idx = joint_cfg["list_joint_idx"]
            if self.intr_sub_que_dict[channel].empty():
                continue

            meas = self.intr_sub_que_dict[channel].get()
            if not isinstance(meas, JointMeasData):
                raise ValueError(
                    f"Expected JointMeasData on '{channel}', got {type(meas)}"
                )
            _, q, qd, _ = meas.get_data()
            with self._meas_lock:
                self._q[idx] = q
                self._qd[idx] = qd

    def _snapshot_meas(self) -> tuple[np.ndarray, np.ndarray]:
        """Thread-safe copy of (q, qd) for the planner."""
        with self._meas_lock:
            return self._q.copy(), self._qd.copy()

    def _publish_plan_result(self, result: PlanResult) -> None:
        """Overwrite with the newest plan (drop stale if control is behind)."""
        try:
            self._plan_result_que.put_nowait(result)
        except Full:
            try:
                self._plan_result_que.get_nowait()
            except Empty:
                pass
            try:
                self._plan_result_que.put_nowait(result)
            except Full:
                pass

    def _consume_latest_plan(self) -> None:
        """Drain the plan queue; apply only the newest result (like ``_new_plan``)."""
        latest: PlanResult | None = None
        while True:
            try:
                latest = self._plan_result_que.get_nowait()
            except Empty:
                break
        if latest is None:
            return

        if latest.state != self._last_logged_state:
            print(
                f"[Time: {self._cur_time}] State changed from "
                f"{self._last_logged_state} to {latest.state}"
            )
            self._last_logged_state = latest.state

        self._ctrl_state = latest.state
        self._qd_des = latest.qd_des.copy()

    # --------------------------------------------------------- FCL / geometry
    def _get_col_links(self, q: np.ndarray) -> dict:
        poses = get_fk_link_poses(
            q, self._pin_model, self._pin_data_plan, self._col_link_names
        )
        jacs = get_link_Jacobians(
            q, self._pin_model, self._pin_data_plan, self._col_link_names
        )

        out = {}
        for i, link_name in enumerate(self._col_link_names):
            pose = poses[i]
            tf = pose @ self._fcl_robot_col_geoms[link_name]["offset"]
            out[link_name] = {
                "col_geom": fcl.CollisionObject(
                    self._fcl_robot_col_geoms[link_name]["fcl_geom"],
                    fcl.Transform(tf[:3, :3], tf[:3, 3]),
                ),
                "p": pose[:3, 3].copy(),
                "R": pose[:3, :3].copy(),
                "J": jacs[i].copy(),
            }
        return out

    def _get_col_obstacles(self) -> dict:
        out = {}
        for obs_name in self._obs_names:
            entry = self._fcl_static_col_geoms[obs_name]
            offset = entry["offset"]
            out[obs_name] = {
                "col_geom": fcl.CollisionObject(
                    entry["fcl_geom"],
                    fcl.Transform(offset[:3, :3], offset[:3, 3]),
                )
            }
        return out

    def _get_col_data(self, dict_col_links: dict, dict_col_obstacles: dict) -> dict:
        collision_data = {name: {} for name in self._col_link_names}

        for link_name, obs_name in self._col_pairs:
            req = fcl.DistanceRequest(enable_nearest_points=True)
            res = fcl.DistanceResult()
            fcl.distance(
                dict_col_links[link_name]["col_geom"],
                dict_col_obstacles[obs_name]["col_geom"],
                req,
                res,
            )

            nearest_p_link = res.nearest_points[0]
            nearest_p_obs = res.nearest_points[1]
            R = dict_col_links[link_name]["R"]
            p = dict_col_links[link_name]["p"]
            J = dict_col_links[link_name]["J"]
            delta = R.T @ (nearest_p_link - p)

            collision_data[link_name][obs_name] = {
                "dist": res.min_distance,
                "nearest_p_link": nearest_p_link,
                "nearest_p_obs": nearest_p_obs,
                "point_Jac": get_point_Jacobian(R, J[3:, :], J[:3, :], delta),
            }
        return collision_data

    # --------------------------------------------------------------- planning
    def _plan_loop(self) -> None:
        """Background loop: FCL distances -> state -> optional OSQP velocity.

        Rate limiting matches the original: gate on ``_cur_time``, and stamp
        ``last_plan_time`` with ``_cur_time`` *after* the plan finishes (so a
        slow QP delays the next plan by a full ``plan_dt``).
        """
        last_plan_time = 0.0

        while not self._stop_event.is_set():
            # Same gate as the original shared-state loop.
            if self._cur_time - last_plan_time > self._plan_dt:
                q, qd = self._snapshot_meas()
                result = self._compute_plan(q, qd)
                self._publish_plan_result(result)
                # Stamp with time at end of planning (original behavior).
                last_plan_time = self._cur_time

            time.sleep(0.0001)

    def _compute_plan(self, q: np.ndarray, qd: np.ndarray) -> PlanResult:
        dict_col_links = self._get_col_links(q)
        dict_col_obstacles = self._get_col_obstacles()
        collision_data = self._get_col_data(dict_col_links, dict_col_obstacles)

        is_healthy = True
        is_nearby = False
        for link_name, obs_name in self._col_pairs:
            col = collision_data[link_name][obs_name]
            delta = col["nearest_p_link"] - col["nearest_p_obs"]
            margin = col["dist"] - self._safety_dist_thr
            col["margin"] = margin

            if margin < 0:
                weight = 0.0
                is_healthy = False
            elif margin < self._margin_thr:
                weight = self._repulsive_coeff * (margin / self._margin_thr - 1.0) ** 2
                is_nearby = True
            else:
                weight = 0.0

            col["col_avoid_vel"] = (
                delta / np.clip(np.linalg.norm(delta), 1e-4, np.inf) * weight
            )

        if not is_healthy:
            print(f"[Time: {self._cur_time}] Robot in collision, stop")
            return PlanResult(
                state="collision",
                qd_des=np.zeros(self._n_x),
                success=False,
            )

        if not is_nearby:
            return PlanResult(
                state="grav_comp",
                qd_des=np.zeros(self._n_x),
                success=True,
            )

        qd_des, success = self._solve_repulsive_qp(q, qd, collision_data)
        return PlanResult(state="repulsive", qd_des=qd_des, success=success)

    def _solve_repulsive_qp(
        self, q: np.ndarray, qd: np.ndarray, collision_data: dict
    ) -> tuple[np.ndarray, bool]:
        list_A, list_lb, list_ub = [], [], []
        for link_name, obs_name in self._col_pairs:
            col = collision_data[link_name][obs_name]
            list_A.append(
                (col["nearest_p_link"] - col["nearest_p_obs"]).T
                @ col["point_Jac"]
                / col["dist"]
            )
            list_lb.append(
                np.array([(-col["dist"] + self._safety_dist_thr) / self._plan_horizon])
            )
            list_ub.append(np.array([1000.0 / self._plan_horizon]))

        A_np = np.vstack(list_A)
        A_lb = np.hstack(list_lb)
        A_ub = np.hstack(list_ub)

        P_np = 0.01 * np.eye(self._n_x)
        g_np = np.zeros(self._n_x)
        for link_name, obs_name in self._col_pairs:
            col = collision_data[link_name][obs_name]
            P_np += 2.0 * col["point_Jac"].T @ col["point_Jac"]
            g_np += -2.0 * col["point_Jac"].T @ col["col_avoid_vel"]

        A = sparse.csc_matrix(np.vstack([np.eye(self._n_x), A_np]))
        l = np.hstack(
            [
                np.maximum(
                    -self._joint_velocity_limit,
                    (self._q_lower_limit - q) / self._plan_horizon,
                ),
                A_lb,
            ]
        )
        u = np.hstack(
            [
                np.minimum(
                    self._joint_velocity_limit,
                    (self._q_upper_limit - q) / self._plan_horizon,
                ),
                A_ub,
            ]
        )

        if np.any(u < l):
            print("u < l, no solution")
            return np.zeros(self._n_x), False

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(P_np),
            q=g_np,
            A=A,
            l=l,
            u=u,
            verbose=False,
        )
        solver.warm_start(x=qd.copy(), y=np.zeros(A.shape[0]))
        res = solver.solve()

        if res.info.status_val == osqp.constant("OSQP_SOLVED"):
            return np.asarray(res.x, dtype=np.float64).copy(), True

        print("Solver status:", res.info.status)
        print(f"[Time: {self._cur_time}] No solution found")
        return np.zeros(self._n_x), False

    # --------------------------------------------------------------- control
    def _update(self) -> None:
        """Apply the latest plan (if any) and publish a torque command."""
        self._consume_latest_plan()

        with self._meas_lock:
            q = self._q.copy()
            qd = self._qd.copy()

        qd_des = self._qd_des.copy()
        q_des = q + self._kp_multiplier * qd_des * self._ctrl_dt

        tau_ff = pin.computeGeneralizedGravity(self._pin_model, self._pin_data_ctrl, q)
        tau_ff[:7] += -0.01 * qd[:7]

        if self._ctrl_state == "grav_comp":
            kp = np.zeros_like(self._kp)
            kd = np.zeros_like(self._kd)
        else:
            kp = self._kp
            kd = self._kd

        cmd = JointCtrlData(num_joints=self._num_joints)
        cmd.set_data(self._cur_time, q_des, qd_des, tau_ff, kp, kd)
        self.ctrl_pub_que.put(cmd)
