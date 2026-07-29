"""IK QP controller: contract a task point to a keyboard-moved 3D target
while respecting linearized FCL collision constraints.

Architecture (same pattern as CollisionAwareGravComp)
----------------------------------------------------
- Control loop at ``ctrl_freq`` (e.g. 1000 Hz): PD / velocity tracking +
  gravity / friction FF + pose publish.
- Planning thread at ``plan_freq`` (e.g. 100 Hz): FK, FCL, contracting
  field, OSQP → ``qd_des``.

Running FCL+OSQP inside the 1 kHz loop made the controller miss ticks,
burst-catch-up, and look like a laggy / freaking target-velocity update.
"""

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
from data_type.basic_types.Pose3DData import Pose3DData
from utils.fcl.getter import get_fcl_geom
from utils.lie.kinematics import get_point_Jacobian
from utils.pinocchio.getter import get_fk_link_pose, get_link_Jacobian


@dataclass
class PlanResult:
    """Latest output from the IK planning thread."""

    qd_des: np.ndarray
    tip: np.ndarray
    velocity: np.ndarray  # tip Cartesian vel implied by qd_des (J @ qd_des)
    mode: str  # "ik" | "home"
    at_goal: bool
    success: bool


class IKQPController(BaseController):
    """Track a 3D target with a contracting velocity field via QP + collision.

    Keyboard (controller terminal)
    ------------------------------
    ``w/s`` ±x, ``a/d`` ±y, ``r/f`` ±z, ``[/]`` step size,
    ``h`` home (joint-space go-to default_q), ``p`` pause, ``q`` quit.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._dict_joints = config["dict_joints"]
        self._num_joints = int(config["num_joints"])
        self._n_x = self._num_joints

        self._robot_col_info_channel = config["sub_manager"]["robot_col_info_channel"]
        self._static_col_info_channel = config["sub_manager"]["static_col_info_channel"]
        self._pose_channel = config["pub_manager"]["pose_channel"]

        ####################################################################
        # Gains (ThreeFingerReactiveGrasping style)
        ####################################################################
        self._est_time_delay = float(config.get("est_time_delay", 0.01))
        self._mean_mass_matrix_diag = np.asarray(
            config["mean_mass_matrix_diag"], dtype=np.float64
        )
        self._kd = np.asarray(config["default_kd"], dtype=np.float64)
        self._kp = self._kd**2 / (4.0 * self._mean_mass_matrix_diag)
        print(f"[IKQP] Computed Kp from default_kd / mean M diag: {self._kp}")

        effective_mass = self._mean_mass_matrix_diag - self._est_time_delay * self._kd
        effective_kd = self._kd - self._kp * self._est_time_delay
        if np.max(effective_mass) < 0:
            raise ValueError(
                f"Effective mass is negative {effective_mass}. Please decrease Kd."
            )
        if np.max(effective_kd) < 0:
            raise ValueError(
                f"Effective kd is negative {effective_kd}. Please decrease Kp."
            )

        self._velocity_control_D_gain = np.asarray(
            config["velocity_control_D_gain"], dtype=np.float64
        )
        self._joint_velocity_limit = np.asarray(
            config["joint_velocity_limit"], dtype=np.float64
        )

        ####################################################################
        # Friction / dynamics feedforward
        ####################################################################
        self._do_grav_comp = bool(config.get("do_grav_comp", True))
        self._do_friction_comp = bool(config.get("do_friction_comp", True))
        self._do_coriolis_comp = bool(config.get("do_coriolis_comp", True))
        self._Fjc = np.asarray(config["Fjc"], dtype=np.float64)
        self._friction_phi = float(config.get("friction_phi", 0.03))

        self._task_link = str(config["task_link"])
        self._task_point_local = np.asarray(
            config.get("task_point_local", [0.0, 0.0, 0.0]), dtype=np.float64
        ).reshape(3)
        self._vel_gain = float(config.get("vel_gain", 1.0))
        self._v_max = float(config.get("v_max", 0.25))
        self._cart_damping = float(config.get("cart_damping", 0.0))
        self._pos_tol = float(config.get("pos_tol", 0.001))
        self._qd_filter = float(config.get("qd_filter", 0.3))
        self._qp_reg = float(config.get("qp_reg", 0.05))
        self._plan_horizon = float(config.get("plan_horizon", 0.1))
        self._safety_dist_thr = float(config.get("safty_dist_thr", 0.02))
        # Max demanded gap-opening speed when already inside the safety margin (m/s).
        self._col_sep_rate_max = float(config.get("col_sep_rate_max", 0.25))
        self._use_position_pd = bool(config.get("use_position_pd", False))

        # Planning rate (FCL + OSQP). Keep well below ctrl_freq.
        self._plan_freq = float(config.get("plan_freq", 100.0))
        self._plan_dt = 1.0 / self._plan_freq

        self._joint_vel_gain = float(config.get("joint_vel_gain", 1.0))
        self._joint_damping = float(config.get("joint_damping", 0.5))
        self._joint_pos_tol = float(config.get("joint_pos_tol", 0.02))

        self._col_link_names = list(config["col_link_names"])
        self._obs_names = list(config["obs_names"])
        self._col_pairs = [list(p) for p in config["col_pairs"]]

        self._target_step = float(config.get("target_step", 0.01))
        self._target_step_min = float(config.get("target_step_min", 0.002))
        self._target_step_max = float(config.get("target_step_max", 0.05))

        pin_robot = pin.RobotWrapper.BuildFromURDF(
            config["urdf_path"], list(config["mesh_path"].values())
        )
        self._pin_model = pin_robot.model
        # Separate Data per thread — Pinocchio Data is not thread-safe.
        self._pin_data_ctrl = pin_robot.data
        self._pin_data_plan = self._pin_model.createData()
        self._q_lower = self._pin_model.lowerPositionLimit.copy()
        self._q_upper = self._pin_model.upperPositionLimit.copy()

        self._q_home = np.asarray(config["default_q_values"], dtype=np.float64).copy()
        self._q = self._q_home.copy()
        self._qd = np.zeros(self._num_joints, dtype=np.float64)

        self._p_des = np.zeros(3, dtype=np.float64)
        self._target_initialized = False
        self._ctrl_mode = "ik"  # "ik" | "home"

        # Held by the control thread; refreshed from the plan-result queue.
        self._qd_des = np.zeros(self._num_joints, dtype=np.float64)
        self._qd_des_filt = np.zeros(self._num_joints, dtype=np.float64)
        self._tip = np.zeros(3, dtype=np.float64)
        self._v_cmd = np.zeros(3, dtype=np.float64)
        self._at_goal = True

        self._meas_lock = threading.Lock()
        self._plan_result_que: Queue[PlanResult] = Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._plan_thread = threading.Thread(
            target=self._plan_loop, name="ik_qp_plan", daemon=False
        )

        self._fcl_robot: dict = {}
        self._fcl_static: dict = {}

        print("[IKQP] Keys: w/s ±x, a/d ±y, r/f ±z, [/] step, h home, p pause, q quit.")
        print(f"[IKQP] ctrl_freq={self._ctrl_freq} Hz, plan_freq={self._plan_freq} Hz")

    # ------------------------------------------------------------------ init
    def initialize(self) -> None:
        self._fcl_robot = self._wait_for_col_geoms(
            self.intr_sub_que_dict[self._robot_col_info_channel], "robot"
        )
        self._fcl_static = self._wait_for_col_geoms(
            self.extr_sub_que_dict[self._static_col_info_channel], "static"
        )
        print("[IKQP] Initialized with FCL geoms.")

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

    # ----------------------------------------------------------- thread hooks
    def _sub_thread_start(self) -> None:
        if not self._plan_thread.is_alive():
            self._plan_thread.start()
            print("[IKQP] Planning thread started.")

    def _sub_thread_stop(self) -> None:
        if self._plan_thread.is_alive():
            self._stop_event.set()
            self._plan_thread.join(timeout=2.0)
            print("[IKQP] Planning thread stopped.")

    # -------------------------------------------------------------- keyboard
    def _enter_ik_mode(self, reason: str = "") -> None:
        with self._meas_lock:
            if self._ctrl_mode == "ik":
                switching = False
            else:
                switching = True
                self._ctrl_mode = "ik"
            q = self._q.copy()

        if not switching and self._target_initialized:
            return

        p, _, _, _ = self._task_point_and_jacobian(q, self._pin_data_ctrl)
        with self._meas_lock:
            self._p_des = p.copy()
            self._target_initialized = True
            self._qd_des_filt[:] = 0.0
        msg = "[IKQP] Mode → ik"
        if reason:
            msg += f" ({reason})"
        print(msg)

    def _enter_home_mode(self) -> None:
        with self._meas_lock:
            self._ctrl_mode = "home"
            self._qd_des_filt[:] = 0.0
        print("[IKQP] Mode → home (joint-space go to default_q)")

    def _handle_key(self, key: str) -> None:
        if key in ("p", "q"):
            super()._handle_key(key)
            return

        if key == "h":
            self._enter_home_mode()
            return

        if key == "[":
            self._target_step = max(self._target_step_min, self._target_step * 0.5)
            print(f"[IKQP] target_step = {self._target_step:.4f} m")
            return
        if key == "]":
            self._target_step = min(self._target_step_max, self._target_step * 2.0)
            print(f"[IKQP] target_step = {self._target_step:.4f} m")
            return

        step = self._target_step
        delta = {
            "w": np.array([step, 0.0, 0.0]),
            "s": np.array([-step, 0.0, 0.0]),
            "a": np.array([0.0, step, 0.0]),
            "d": np.array([0.0, -step, 0.0]),
            "r": np.array([0.0, 0.0, step]),
            "f": np.array([0.0, 0.0, -step]),
        }.get(key)

        if delta is None:
            print(f"[IKQP] Unused key: {key!r}")
            return

        self._enter_ik_mode("cartesian nudge")
        with self._meas_lock:
            self._p_des = self._p_des + delta
            p_des = self._p_des.copy()
        print(f"[IKQP] p_des = [{p_des[0]:.3f}, {p_des[1]:.3f}, {p_des[2]:.3f}]")

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

    def _snapshot_for_plan(self):
        """Thread-safe snapshot for the planner."""
        with self._meas_lock:
            return (
                self._q.copy(),
                self._qd.copy(),
                self._p_des.copy(),
                self._ctrl_mode,
                self._target_initialized,
            )

    def _publish_plan_result(self, result: PlanResult) -> None:
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
        latest: PlanResult | None = None
        while True:
            try:
                latest = self._plan_result_que.get_nowait()
            except Empty:
                break
        if latest is None:
            return
        self._qd_des = latest.qd_des.copy()
        self._tip = latest.tip.copy()
        self._v_cmd = latest.velocity.copy()
        self._at_goal = latest.at_goal

    # --------------------------------------------------------------- geometry
    def _task_point_and_jacobian(self, q: np.ndarray, pin_data):
        T = get_fk_link_pose(q, self._pin_model, pin_data, self._task_link)
        R = T[:3, :3]
        p_link = T[:3, 3]
        p = p_link + R @ self._task_point_local
        J = get_link_Jacobian(q, self._pin_model, pin_data, self._task_link)
        J_R, J_p_origin = J[:3, :], J[3:, :]
        J_p = get_point_Jacobian(R, J_p_origin, J_R, self._task_point_local)
        return p, J_p, R, p_link

    def _get_col_links(self, q: np.ndarray) -> dict:
        out = {}
        for link_name in self._col_link_names:
            T = get_fk_link_pose(q, self._pin_model, self._pin_data_plan, link_name)
            entry = self._fcl_robot[link_name]
            tf = T @ entry["offset"]
            out[link_name] = {
                "col_geom": fcl.CollisionObject(
                    entry["fcl_geom"],
                    fcl.Transform(tf[:3, :3], tf[:3, 3]),
                ),
                "p": T[:3, 3].copy(),
                "R": T[:3, :3].copy(),
                "J": get_link_Jacobian(
                    q, self._pin_model, self._pin_data_plan, link_name
                ),
            }
        return out

    def _get_col_obstacles(self) -> dict:
        out = {}
        for name in self._obs_names:
            entry = self._fcl_static[name]
            T = entry["offset"]
            out[name] = {
                "col_geom": fcl.CollisionObject(
                    entry["fcl_geom"],
                    fcl.Transform(T[:3, :3], T[:3, 3]),
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
                "dist": float(res.min_distance),
                "nearest_p_link": nearest_p_link,
                "nearest_p_obs": nearest_p_obs,
                "point_Jac": get_point_Jacobian(R, J[3:, :], J[:3, :], delta),
            }
        return collision_data

    # -------------------------------------------------------------------- QP
    def _contracting_velocity(
        self, p: np.ndarray, p_dot: np.ndarray, p_des: np.ndarray
    ):
        err = p_des - p
        if float(np.linalg.norm(err)) < self._pos_tol:
            return np.zeros(3, dtype=np.float64)
        v = self._vel_gain * err - self._cart_damping * p_dot
        n = float(np.linalg.norm(v))
        if n > self._v_max and n > 1e-12:
            v = v * (self._v_max / n)
        return v

    # -------------------------------------------------------------------- QP
    def _joint_velocity_bounds(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feasible joint-velocity box from limits + position limits over horizon.

        Heuristic: evaluate position-limit rates from ``q`` clipped into the
        joint range so measurement overshoot cannot create ``l > u``.
        """
        T = max(self._plan_horizon, 1e-4)
        q_clip = np.clip(q, self._q_lower, self._q_upper)
        l = np.maximum(-self._joint_velocity_limit, (self._q_lower - q_clip) / T)
        u = np.minimum(self._joint_velocity_limit, (self._q_upper - q_clip) / T)
        # Final guard (should be rare).
        bad = u < l
        if np.any(bad):
            mid = 0.5 * (l + u)
            l = np.where(bad, mid, l)
            u = np.where(bad, mid, u)
        return l, u

    def _collision_rows(
        self, collision_data: dict
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Linearized collision inequalities with a capped escape rate.

        When ``d < d_safe`` the hard QP would demand
        ``(-d + d_safe) / T``, which can be huge / infeasible. Cap that
        demanded separating speed at ``col_sep_rate_max``.
        """
        T = max(self._plan_horizon, 1e-4)
        list_A, list_l, list_u = [], [], []
        for link_name, obs_name in self._col_pairs:
            col = collision_data[link_name][obs_name]
            d = max(float(col["dist"]), 1e-6)
            delta = col["nearest_p_link"] - col["nearest_p_obs"]
            a = ((delta / d) @ col["point_Jac"]).reshape(1, -1)
            # Nominal: do not close the gap faster than reaching d_safe in T.
            sep = (-d + self._safety_dist_thr) / T
            if sep > self._col_sep_rate_max:
                sep = self._col_sep_rate_max
            list_A.append(a)
            list_l.append(np.array([sep], dtype=np.float64))
            list_u.append(np.array([1000.0 / T], dtype=np.float64))
        return list_A, list_l, list_u

    def _osqp_solve(
        self,
        P: np.ndarray,
        g: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray | None:
        # Ensure bound consistency after stacking.
        bad = u < l
        if np.any(bad):
            mid = 0.5 * (l + u)
            l = l.copy()
            u = u.copy()
            l[bad] = mid[bad]
            u[bad] = mid[bad]

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(P),
            q=g,
            A=sparse.csc_matrix(A),
            l=l,
            u=u,
            verbose=False,
            polish=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
        )
        res = solver.solve()
        if res.info.status_val in (
            osqp.constant("OSQP_SOLVED"),
            osqp.constant("OSQP_SOLVED_INACCURATE"),
        ):
            return np.asarray(res.x, dtype=np.float64).copy()
        return None

    def _solve_ik_qp(
        self, q: np.ndarray, J_p: np.ndarray, v_star: np.ndarray, collision_data: dict
    ) -> np.ndarray:
        """Solve IK QP with joint + collision constraints.

        If infeasible (e.g. large ``v*`` conflicts with ``d > d_safe``), return
        ``qd = 0`` — never drop collision constraints or otherwise command
        motion that may close the gap.
        """
        n = self._n_x
        P = 2.0 * (J_p.T @ J_p) + self._qp_reg * np.eye(n)
        g = -2.0 * (J_p.T @ v_star)
        l_j, u_j = self._joint_velocity_bounds(q)

        col_A, col_l, col_u = self._collision_rows(collision_data)
        if col_A:
            A = np.vstack([np.eye(n), *col_A])
            l = np.hstack([l_j, *col_l])
            u = np.hstack([u_j, *col_u])
        else:
            A = np.eye(n)
            l, u = l_j, u_j

        qd = self._osqp_solve(P, g, A, l, u)
        if qd is not None:
            return qd

        print("[IKQP] QP infeasible under collision constraints — qd_des = 0")
        return np.zeros(n, dtype=np.float64)

    def _joint_space_home_velocity(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        err = self._q_home - q
        if float(np.linalg.norm(err, ord=np.inf)) < self._joint_pos_tol:
            return np.zeros(self._n_x, dtype=np.float64)
        qd_star = self._joint_vel_gain * err - self._joint_damping * qd
        return np.clip(qd_star, -self._joint_velocity_limit, self._joint_velocity_limit)

    # --------------------------------------------------------------- planning
    def _plan_loop(self) -> None:
        last_plan_time = 0.0
        while not self._stop_event.is_set():
            if self._cur_time - last_plan_time > self._plan_dt:
                result = self._compute_plan()
                self._publish_plan_result(result)
                last_plan_time = self._cur_time
            time.sleep(0.0001)

    def _compute_plan(self) -> PlanResult:
        q, qd, p_des, mode, target_init = self._snapshot_for_plan()
        p, J_p, _, _ = self._task_point_and_jacobian(q, self._pin_data_plan)

        if mode == "home":
            err = self._q_home - q
            at_goal = float(np.linalg.norm(err, ord=np.inf)) < self._joint_pos_tol
            qd_des = (
                np.zeros(self._n_x)
                if at_goal
                else self._joint_space_home_velocity(q, qd)
            )
            return PlanResult(
                qd_des=qd_des,
                tip=p,
                velocity=J_p @ qd_des,
                mode="home",
                at_goal=at_goal,
                success=True,
            )

        # IK mode
        if not target_init:
            with self._meas_lock:
                self._p_des = p.copy()
                self._target_initialized = True
                p_des = p.copy()
            print(
                f"[IKQP] Target seeded at fingertip "
                f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
            )

        err = p_des - p
        at_goal = float(np.linalg.norm(err)) < self._pos_tol
        if at_goal:
            return PlanResult(
                qd_des=np.zeros(self._n_x),
                tip=p,
                velocity=np.zeros(3),
                mode="ik",
                at_goal=True,
                success=True,
            )

        p_dot = J_p @ qd
        v_star = self._contracting_velocity(p, p_dot, p_des)
        dict_col_links = self._get_col_links(q)
        dict_col_obstacles = self._get_col_obstacles()
        collision_data = self._get_col_data(dict_col_links, dict_col_obstacles)

        qd_des = self._solve_ik_qp(q, J_p, v_star, collision_data)
        return PlanResult(
            qd_des=qd_des,
            tip=p,
            velocity=J_p @ qd_des,
            mode="ik",
            at_goal=False,
            success=True,
        )

    # ---------------------------------------------------------------- feedforward
    def _friction_compensation(self, qd_des: np.ndarray) -> np.ndarray:
        if not self._do_friction_comp:
            return np.zeros(self._n_x, dtype=np.float64)
        phi = max(self._friction_phi, 1e-6)
        return self._Fjc * np.clip(qd_des / phi, -1.0, 1.0)

    def _tau_ff(self, q: np.ndarray, qd: np.ndarray, qd_des: np.ndarray) -> np.ndarray:
        if self._do_grav_comp:
            tau = pin.computeGeneralizedGravity(self._pin_model, self._pin_data_ctrl, q)
        else:
            tau = np.zeros(self._n_x, dtype=np.float64)

        if self._do_coriolis_comp:
            pin.computeCoriolisMatrix(self._pin_model, self._pin_data_ctrl, q, qd)
            tau = tau + self._pin_data_ctrl.C @ qd

        return tau + self._friction_compensation(qd_des)

    def _publish_target(self, tip: np.ndarray, velocity: np.ndarray) -> None:
        with self._meas_lock:
            p_des = self._p_des.copy()
        pose = Pose3DData(name=self._pose_channel)
        pose.set_data(self._cur_time, p_des, velocity, tip)
        que = self.pub_que_dict.get(self._pose_channel)
        if que is None:
            return
        try:
            que.put_nowait(pose)
        except Full:
            try:
                que.get_nowait()
            except Empty:
                pass
            try:
                que.put_nowait(pose)
            except Full:
                pass

    # ---------------------------------------------------------------- update
    def _update(self) -> None:
        """Fast control tick: apply latest plan, do NOT run FCL/OSQP here."""
        self._consume_latest_plan()

        with self._meas_lock:
            q = self._q.copy()
            qd = self._qd.copy()
            mode = self._ctrl_mode

        # Low-pass joint velocity command on the control thread.
        a = float(np.clip(self._qd_filter, 0.0, 1.0))
        self._qd_des_filt = a * self._qd_des + (1.0 - a) * self._qd_des_filt
        qd_des = self._qd_des_filt

        if self._at_goal or not np.any(np.abs(self._qd_des) > 1e-9):
            if mode == "home" and self._at_goal:
                q_des = self._q_home
            else:
                q_des = q
            qd_des = np.zeros(self._n_x)
            self._qd_des_filt[:] = 0.0
            kp, kd = self._kp, self._kd
        elif self._use_position_pd:
            q_des = q + qd_des * self._ctrl_dt
            kp, kd = self._kp, self._kd
        else:
            q_des = q
            kp = np.zeros_like(self._kp)
            kd = self._velocity_control_D_gain

        tau_ff = self._tau_ff(q, qd, qd_des)
        cmd = JointCtrlData(num_joints=self._num_joints)
        cmd.set_data(self._cur_time, q_des, qd_des, tau_ff, kp, kd)
        self.ctrl_pub_que.put(cmd)
        self._publish_target(tip=self._tip, velocity=self._v_cmd)
