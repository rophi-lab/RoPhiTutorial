"""Three-finger reactive grasping for Flexiv Rizon + Robotis RH-5.

Ported from Manipulator-Software
``controller/brl_arm_5F_hand/grasping/ThreeFingerReactiveGrasping.py``.

Reaching uses the brl ``_plan_reaching`` stack:
heuristic collision fingertip LVF → tip orientation VF → gripper VF →
weighted IKQP with FCL env constraints + PredefinedObj object constraints.

Closing: squeeze (tip attractors + hand-close τ) for ``lift_delay_s``, then
arm joints to ``default_q`` while holding the squeeze.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Empty, Queue

import fcl
import numpy as np
import osqp
import pinocchio as pin
from omegaconf import DictConfig
from scipy import sparse

from assets.object_mesh import get_grasp_points_in_cad, get_nominal_pose_to_cad
from controller.BaseController import BaseController
from data_type.basic_types.ColInfoData import ColInfoData
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData
from data_type.basic_types.NamedVecListData import NamedVecListData
from data_type.basic_types.SE3PoseData import SE3PoseData
from utils.fcl.getter import get_fcl_geom
from utils.lie.kinematics import get_point_Jacobian
from utils.pinocchio.getter import (
    get_fk_link_pose,
    get_fk_link_poses,
    get_link_Jacobians,
)
from utils.planning.heuristics import heuristic_path_initialization_general
from utils.planning.path_sqp import smooth_multi_finger_paths_sqp
from utils.shape_primitives.PredefinedObj import PredefinedObj
from utils.velocity_fields.grasping import goto_x_des_const_then_linear_clamped
from utils.velocity_fields.weight_func import (
    speed_const_then_linear_clamped,
    weight_func_tanh,
)

DIST_THR_REACHING2CLOSING_GRIPPER = 0.02
DIST_THR_ANY2REACHING = 0.05
DEFAULT_V = 1.5
DEFAULT_EPS = 0.25

NUM_POINTS = 100
COLLISION_MARGIN_TO_OBJECT_QP = 0.01
COLLISION_MARGIN_TO_OBJECT_LVF = 0.01
LINEAR_VELOCITY_FIELD_V = 1.0
LINEAR_VELOCITY_FIELD_EPS = 0.1
HEURISTIC_VIA_LENGTH = 0.07

OMEGA_INDEX_GAIN = 1.0
OMEGA_MIDDLE_GAIN = 1.0
GRIPPER_NOMINAL_V = 1.0
GRIPPER_NOMINAL_EPS = 0.1

# Finger phalanx / tip frames used like brl thumb2..4 + ft (RH-5 names).
_FINGER_FK_FRAMES = [
    "finger_r_link_1_thumb2",
    "finger_r_link_1_thumb3",
    "finger_r_link_1_thumb4",
    "thumb_tip",
    "finger_r_link_2_index2",
    "finger_r_link_2_index3",
    "finger_r_link_2_index4",
    "index_tip",
    "finger_r_link_3_middle2",
    "finger_r_link_3_middle3",
    "finger_r_link_3_middle4",
    "middle_tip",
    "finger_r_link_4_ring2",
    "finger_r_link_4_ring3",
    "finger_r_link_4_ring4",
    "finger_r_link_5_little2",
    "finger_r_link_5_little3",
    "finger_r_link_5_little4",
]


class ThreeFingerReactiveGrasping(BaseController):
    """Reactive 3-finger grasp with brl-style collision-aware reaching QP."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._dict_joints = config["dict_joints"]
        self._n = int(config["num_joints"])
        self._n_x = self._n
        self._num_arm = int(config.get("num_arm_joints", 7))
        self._obj_name = str(config.get("obj_name", "green_bowl"))
        self._object_pose_channel = config["sub_manager"].get(
            "object_pose_channel", "sw_grasp_object_pose"
        )
        self._robot_col_info_channel = config["sub_manager"]["robot_col_info_channel"]
        self._static_col_info_channel = config["sub_manager"]["static_col_info_channel"]
        self._grasp_candidates_channel = config["pub_manager"].get(
            "grasp_candidates_channel", ""
        )
        self._fingertip_viz_channel = config["pub_manager"].get(
            "fingertip_viz_channel", ""
        )
        self._ft_path_viz_channel = config["pub_manager"].get("ft_path_viz_channel", "")
        self._contact_force_arrows_channel = config["pub_manager"].get(
            "contact_force_arrows_channel", ""
        )
        self._contact_normal_arrows_channel = config["pub_manager"].get(
            "contact_normal_arrows_channel", ""
        )
        self._viz_stride = int(config.get("viz_publish_stride", 20))
        self._viz_tick = 0

        self._col_link_names = list(
            config.get(
                "col_link_names",
                ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "palm"],
            )
        )
        self._obs_names = list(
            config.get("obs_names", ["floor", "wall_back", "wall_left"])
        )
        self._col_pairs = [list(p) for p in config.get("col_pairs", [])]
        self._plan_horizon = float(config.get("plan_horizon", 0.05))
        self._col_margin_obj_qp = float(
            config.get("collision_margin_to_object_qp", COLLISION_MARGIN_TO_OBJECT_QP)
        )
        self._col_margin_lvf = float(
            config.get("collision_margin_to_object_lvf", COLLISION_MARGIN_TO_OBJECT_LVF)
        )
        self._num_lvf_points = int(config.get("num_lvf_points", NUM_POINTS))
        self._num_path_sqp_iterations = int(config.get("num_path_sqp_iterations", 3))
        self._path_sqp_savgol_window = int(config.get("path_sqp_savgol_window", 5))
        self._lvf_v = float(
            config.get("linear_velocity_field_v", LINEAR_VELOCITY_FIELD_V)
        )
        self._lvf_eps = float(
            config.get("linear_velocity_field_eps", LINEAR_VELOCITY_FIELD_EPS)
        )
        self._heuristic_via_length = float(
            config.get("heuristic_via_length", HEURISTIC_VIA_LENGTH)
        )
        self._env_dist_margin = float(config.get("env_collision_margin", 0.05))
        self._lift_delay_s = float(config.get("lift_delay_s", 2.0))
        self._lift_speed = float(config.get("lift_speed", 0.8))
        self._lift_arm_eps = float(config.get("lift_arm_eps", 0.08))
        self._squeeze_width_multiple = float(config.get("squeeze_width_multiple", 0.55))
        self._closing_t0 = None
        self._lift_active = False
        self._lift_done = False
        self._palm_frame = str(config.get("palm_frame", "palm"))

        pin_robot = pin.RobotWrapper.BuildFromURDF(
            config["urdf_path"], list(config["mesh_path"].values())
        )
        self._pin_model = pin_robot.model
        self._pin_data = pin_robot.data
        self._pin_data_plan = self._pin_model.createData()

        self._fk_frame_names = list(
            dict.fromkeys(_FINGER_FK_FRAMES + self._col_link_names + [self._palm_frame])
        )
        for name in self._fk_frame_names:
            if not self._pin_model.existFrame(name):
                raise ValueError(f"frame {name!r} missing in URDF")

        self._q_lower = self._pin_model.lowerPositionLimit.copy()
        self._q_upper = self._pin_model.upperPositionLimit.copy()
        margin = float(config.get("joint_limit_margin", 0.05))
        self._q_lower += margin
        self._q_upper -= margin

        self._q_nominal = np.asarray(
            config["default_q_values"], dtype=np.float64
        ).copy()
        self.q_nominal_gripper_pose = np.asarray(
            config.get("q_nominal_gripper", self._q_nominal[self._num_arm :]),
            dtype=np.float64,
        ).copy()
        self._q_hand_open = np.asarray(
            config.get("q_hand_open", np.zeros(20)), dtype=np.float64
        ).copy()
        self._q_hand_close = np.asarray(
            config.get("q_hand_close", self.q_nominal_gripper_pose),
            dtype=np.float64,
        ).copy()

        self._kd = np.asarray(config["default_kd"], dtype=np.float64)
        mass = np.asarray(config["mean_mass_matrix_diag"], dtype=np.float64)
        self._kp = self._kd**2 / (4.0 * np.maximum(mass, 1e-6))
        self._vel_kd = np.asarray(
            config.get("velocity_control_D_gain", self._kd), dtype=np.float64
        )
        self._joint_velocity_limit = np.asarray(
            config.get("joint_velocity_limit", np.full(self._n, 2.0)),
            dtype=np.float64,
        )
        # IKQP ||qd||^2 regularization (larger → more conservative joint motion).
        # Prefer full-vector ``ikqp_qd_reg``; else ``ikqp_reg_arm`` (scalar or
        # per-arm-joint list) + ``ikqp_reg_hand`` (scalar or per-hand list).
        if "ikqp_qd_reg" in config:
            self._ikqp_qd_reg = np.asarray(config["ikqp_qd_reg"], dtype=np.float64)
            if self._ikqp_qd_reg.shape != (self._n,):
                raise ValueError(
                    f"ikqp_qd_reg must have length {self._n}, got {self._ikqp_qd_reg.shape}"
                )
        else:
            reg_arm = np.asarray(
                config.get("ikqp_reg_arm", 10.0), dtype=np.float64
            ).reshape(-1)
            if reg_arm.size == 1:
                reg_arm = np.full(self._num_arm, float(reg_arm[0]), dtype=np.float64)
            elif reg_arm.size != self._num_arm:
                raise ValueError(
                    f"ikqp_reg_arm must be a scalar or length-{self._num_arm} list, "
                    f"got shape {reg_arm.shape}"
                )
            n_hand = self._n - self._num_arm
            reg_hand = np.asarray(
                config.get("ikqp_reg_hand", 0.01), dtype=np.float64
            ).reshape(-1)
            if reg_hand.size == 1:
                reg_hand = np.full(n_hand, float(reg_hand[0]), dtype=np.float64)
            elif reg_hand.size != n_hand:
                raise ValueError(
                    f"ikqp_reg_hand must be a scalar or length-{n_hand} list, "
                    f"got shape {reg_hand.shape}"
                )
            self._ikqp_qd_reg = np.concatenate([reg_arm, reg_hand])
        # Penalize ||qd - qd_prev||^2 for temporal smoothness (0 disables).
        self._ikqp_reg_qd_rate = float(config.get("ikqp_reg_qd_rate", 1.0))
        # Cap collision escape-rate demands so deep penetration cannot make
        # the QP primal-infeasible (OSQP then leaves a ~1e9 garbage ``x``).
        self._ikqp_max_escape_rate = float(config.get("ikqp_max_escape_rate", 0.35))
        self._do_grav_comp = bool(config.get("do_grav_comp", True))
        self._Fjc = np.asarray(config.get("Fjc", np.zeros(self._n)), dtype=np.float64)
        self._do_friction_comp = bool(config.get("do_friction_comp", True))

        gps = get_grasp_points_in_cad(self._obj_name)
        self._grasp_points_local = np.concatenate([gps, gps[:, ::-1, :]], axis=0)
        T_n2c = get_nominal_pose_to_cad(self._obj_name)
        self._grasp_reaching_direction_cad = T_n2c[:3, :3] @ np.array(
            [1.0, 0.0, 0.0], dtype=np.float64
        )

        obj_mesh = Path(
            str(
                config.get(
                    "obj_mesh_path",
                    f"assets/object_mesh/{self._obj_name}/textured_mesh.obj",
                )
            )
        )
        if not obj_mesh.exists():
            raise FileNotFoundError(f"Object mesh missing: {obj_mesh}")
        self._target_obj = PredefinedObj(str(obj_mesh))

        self._q = self._q_nominal.copy()
        self._qd = np.zeros(self._n, dtype=np.float64)
        self._T_obj = np.eye(4, dtype=np.float64)
        self._have_obj = False
        self._state = "grav_comp"

        self._qd_des = np.zeros(self._n, dtype=np.float64)
        self._tau_ff_des = np.zeros(self._n, dtype=np.float64)
        # Once IKQP fails we zero qd for that plan only (BRL strategy) — no
        # permanent latch. Kept for API compat with subclasses / keys.
        self._ikqp_fault = False
        self._ikqp_fail_log_t = 0.0
        self._plan_results: Queue = Queue(maxsize=1)
        self._stop_plan = threading.Event()
        self._plan_dt = 1.0 / float(config.get("plan_freq", 50))
        self._plan_thread = threading.Thread(
            target=self._plan_loop, name="grasp_plan", daemon=True
        )

        self._fcl_static_col_geoms: dict = {}
        self._fcl_robot_col_geoms: dict = {}

        self._ft_x_thumb = np.zeros(3)
        self._ft_x_index = np.zeros(3)
        self._ft_x_middle = np.zeros(3)
        self._ft_n_thumb = np.zeros(3)
        self._ft_n_index = np.zeros(3)
        self._ft_n_middle = np.zeros(3)
        self._ft_x_thumb_target = np.zeros(3)
        self._ft_x_index_target = np.zeros(3)
        self._ft_x_middle_target = np.zeros(3)
        self._ft_n_index_target = np.zeros(3)
        self._ft_n_middle_target = np.zeros(3)
        self._ft_targets = np.zeros((3, 3), dtype=np.float64)
        self._ft_targets_valid = False
        self._ft_paths = np.zeros((3, self._num_lvf_points + 2, 3), dtype=np.float64)
        self._error_reaching = 0.0
        self._error_grasp = 0.0
        self._error_index_angle = 0.0
        self._error_middle_angle = 0.0
        self._active_grasp_idx = 0
        # Sticky grasp selection: re-argmin every plan tick flips between
        # antipodal / flipped pairs and makes fingertip targets jump.
        self._grasp_selection_locked = False
        self._index_middle_assignment = None  # 0 or 1 once locked
        self._grasp_switch_margin = float(config.get("grasp_switch_margin", 0.03))
        self._grasp_reach_dir_tf = self._grasp_reaching_direction_cad.copy()

        # Placeholder poses / Jacobians filled in ``_compute_variables``.
        self._poses: dict[str, np.ndarray] = {}
        self._J_x: dict[str, np.ndarray] = {}
        self._J_R: dict[str, np.ndarray] = {}
        self._J_ft_x_thumb = np.zeros((3, self._n))
        self._J_ft_x_index = np.zeros((3, self._n))
        self._J_ft_x_middle = np.zeros((3, self._n))
        self._J_q_gripper = np.hstack([np.zeros((20, 7)), np.eye(20)])

        print(
            f"[3FGrasp] obj={self._obj_name}. "
            "Keys: g=grav, d=default, v=reach/grasp, c=close, p/q"
        )

    def initialize(self) -> None:
        self._fcl_robot_col_geoms = self._wait_for_col_geoms(
            self.intr_sub_que_dict[self._robot_col_info_channel], "robot"
        )
        self._fcl_static_col_geoms = self._wait_for_col_geoms(
            self.extr_sub_que_dict[self._static_col_info_channel], "static"
        )
        self._plan_thread.start()
        print("[3FGrasp] Initialized with FCL geoms; planning thread running.")

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
                raise TimeoutError(f"Timeout waiting for {label} ColInfo.")
            time.sleep(0.001)

    def stop(self) -> None:
        self._stop_plan.set()
        self._finished = True
        print("[3FGrasp] Stopped planning thread.")
        super().stop()

    def _enter_closing(self) -> None:
        self._state = "closing_gripper"
        self._closing_t0 = time.time()
        self._lift_active = False
        self._lift_done = False
        print(
            f"[3FGrasp] → closing_gripper "
            f"(squeeze {self._lift_delay_s:.1f}s, then arm → default_q)"
        )

    def _reset_closing_flags(self) -> None:
        self._closing_t0 = None
        self._lift_active = False
        self._lift_done = False

    def _clear_ikqp_fault(self) -> None:
        """Compatibility no-op (IKQP failures no longer latch)."""
        self._ikqp_fault = False

    def _ikqp_fail_zeros(self, reason: str) -> np.ndarray:
        """BRL strategy: failed / infeasible IKQP → qd=0 this plan only (no latch).

        Never command OSQP's primal-infeasibility certificate in ``res.x``.
        """
        now = time.time()
        if now - getattr(self, "_ikqp_fail_log_t", 0.0) >= 1.0:
            self._ikqp_fail_log_t = now
            print(f"[3FGrasp] IKQP no solution ({reason}) → qd=0 this plan")
        return np.zeros(self._n_x, dtype=np.float64)

    def _handle_key(self, key: str) -> None:
        if key in ("p", "q"):
            super()._handle_key(key)
            return
        if key == "g":
            self._clear_ikqp_fault()
            self._state = "grav_comp"
            self._reset_closing_flags()
            print("[3FGrasp] → grav_comp")
            return
        if key == "d":
            self._clear_ikqp_fault()
            self._state = "default"
            self._reset_closing_flags()
            print("[3FGrasp] → default (home)")
            return
        if key == "v":
            if not self._have_obj:
                print("[3FGrasp] No object pose yet — wait for env publish.")
                return
            self._clear_ikqp_fault()
            self._enter_reaching()
            print("[3FGrasp] → reaching")
            return
        if key == "c":
            if not self._have_obj:
                print("[3FGrasp] No object pose yet — wait for env publish.")
                return
            self._clear_ikqp_fault()
            self._enter_closing()
            return
        print(f"[3FGrasp] Unused key: {key!r}")

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
            self._q[idx] = q
            self._qd[idx] = qd

        ch = self._object_pose_channel
        if ch and ch in self.extr_sub_que_dict:
            data = None
            while not self.extr_sub_que_dict[ch].empty():
                data = self.extr_sub_que_dict[ch].get()
            if isinstance(data, SE3PoseData):
                R = _quat_wxyz_to_R(data.quat_wxyz)
                self._T_obj[:3, :3] = R
                self._T_obj[:3, 3] = data.position
                self._have_obj = True
                self._target_obj.set_pose(data.position, R)

        for ch in (self._static_col_info_channel,):
            if ch in self.extr_sub_que_dict:
                while not self.extr_sub_que_dict[ch].empty():
                    _ = self.extr_sub_que_dict[ch].get()
        if self._robot_col_info_channel in self.intr_sub_que_dict:
            while not self.intr_sub_que_dict[self._robot_col_info_channel].empty():
                _ = self.intr_sub_que_dict[self._robot_col_info_channel].get()

    def _grasp_points_world(self) -> np.ndarray:
        R = self._T_obj[:3, :3]
        p = self._T_obj[:3, 3]
        return self._grasp_points_local @ R.T + p.reshape(1, 1, 3)

    def _compute_variables(self) -> None:
        q = self._q.copy()
        poses = get_fk_link_poses(
            q, self._pin_model, self._pin_data_plan, self._fk_frame_names
        )
        jacs = get_link_Jacobians(
            q, self._pin_model, self._pin_data_plan, self._fk_frame_names
        )
        for name, T, J in zip(self._fk_frame_names, poses, jacs):
            self._poses[name] = T
            self._J_x[name] = J[3:, :]
            self._J_R[name] = J[:3, :]

        self._J_ft_x_thumb = self._J_x["thumb_tip"]
        self._J_ft_x_index = self._J_x["index_tip"]
        self._J_ft_x_middle = self._J_x["middle_tip"]

        tip_n_local = np.array([1.0, 0.0, 0.0])
        self._ft_x_thumb = self._poses["thumb_tip"][:3, 3].copy()
        self._ft_x_index = self._poses["index_tip"][:3, 3].copy()
        self._ft_x_middle = self._poses["middle_tip"][:3, 3].copy()
        self._ft_n_thumb = self._poses["thumb_tip"][:3, :3] @ tip_n_local
        self._ft_n_index = self._poses["index_tip"][:3, :3] @ tip_n_local
        self._ft_n_middle = self._poses["middle_tip"][:3, :3] @ tip_n_local

        self._grasp_reach_dir_tf = (
            self._T_obj[:3, :3] @ self._grasp_reaching_direction_cad
        )
        width_mult = (
            self._squeeze_width_multiple if self._state == "closing_gripper" else 2.0
        )
        (
            self._ft_x_thumb_target,
            self._ft_x_index_target,
            self._ft_x_middle_target,
        ) = self._compute_target_grasp_data(
            self._ft_x_thumb,
            self._ft_x_index,
            self._ft_x_middle,
            grasp_width_multiple=width_mult,
            update_selection=True,
        )
        self._ft_targets[0] = self._ft_x_thumb_target
        self._ft_targets[1] = self._ft_x_index_target
        self._ft_targets[2] = self._ft_x_middle_target
        self._ft_targets_valid = True

        n_i = self._ft_x_thumb_target - self._ft_x_index_target
        self._ft_n_index_target = n_i / (np.linalg.norm(n_i) + 1e-9)
        n_m = self._ft_x_thumb_target - self._ft_x_middle_target
        self._ft_n_middle_target = n_m / (np.linalg.norm(n_m) + 1e-9)

        ft_mid = 0.5 * (self._ft_x_thumb + 0.5 * (self._ft_x_index + self._ft_x_middle))
        ft_mid_t = 0.5 * (
            self._ft_x_thumb_target
            + 0.5 * (self._ft_x_index_target + self._ft_x_middle_target)
        )
        self._error_reaching = float(np.linalg.norm(ft_mid - ft_mid_t))
        # Contact-width mid error for lose-grasp (do not touch active selection).
        thumb_c, index_c, middle_c = self._compute_target_grasp_data(
            self._ft_x_thumb,
            self._ft_x_index,
            self._ft_x_middle,
            grasp_width_multiple=1.0,
            update_selection=False,
        )
        ft_mid_c = 0.5 * (thumb_c + 0.5 * (index_c + middle_c))
        self._error_grasp = float(np.linalg.norm(ft_mid - ft_mid_c))
        self._error_index_angle = float(
            np.arccos(
                np.clip(np.dot(self._ft_n_index, self._ft_n_index_target), -1.0, 1.0)
            )
        )
        self._error_middle_angle = float(
            np.arccos(
                np.clip(np.dot(self._ft_n_middle, self._ft_n_middle_target), -1.0, 1.0)
            )
        )

    def _compute_target_grasp_data(
        self,
        ft_x_thumb,
        ft_x_index,
        ft_x_middle,
        grasp_width_multiple=1.0,
        update_selection=True,
    ):
        """Select antipodal pair and split right contact into index/middle (brl).

        When ``update_selection`` is True, grasp index / index–middle assignment
        are sticky with hysteresis so targets do not flicker every plan tick.
        """
        gps = self._grasp_points_world()
        x_lft = ft_x_thumb
        x_rft = 0.5 * (ft_x_index + ft_x_middle)
        surf_point1 = gps[:, 0, :]
        surf_point2 = gps[:, 1, :]
        cand_center = 0.5 * (surf_point1 + surf_point2)
        cand_diff = 0.5 * (surf_point1 - surf_point2)

        e_lft = np.linalg.norm(x_lft.reshape(1, 3) - surf_point1, axis=1)
        e_rft = np.linalg.norm(x_rft.reshape(1, 3) - surf_point2, axis=1)
        x_diff_curr = 0.5 * (x_lft - x_rft)
        x_diff_curr_u = x_diff_curr / (np.linalg.norm(x_diff_curr) + 1e-9)
        cand_diff_u = cand_diff / (
            np.linalg.norm(cand_diff, axis=1, keepdims=True) + 1e-9
        )
        e_ang = np.arccos(
            np.clip((x_diff_curr_u.reshape(1, 3) * cand_diff_u).sum(axis=1), -1.0, 1.0)
        )
        costs = e_lft + e_rft + 0.1 * e_ang
        best_idx = int(np.argmin(costs))

        if update_selection:
            if (not self._grasp_selection_locked) or (
                self._active_grasp_idx < 0 or self._active_grasp_idx >= costs.shape[0]
            ):
                grasp_idx = best_idx
                self._grasp_selection_locked = True
            else:
                cur_cost = float(costs[self._active_grasp_idx])
                best_cost = float(costs[best_idx])
                # Only switch if another candidate is clearly better.
                if best_cost < cur_cost - self._grasp_switch_margin:
                    grasp_idx = best_idx
                else:
                    grasp_idx = int(self._active_grasp_idx)
            self._active_grasp_idx = grasp_idx
        else:
            grasp_idx = int(
                np.clip(self._active_grasp_idx, 0, max(costs.shape[0] - 1, 0))
            )

        x_ft_center = cand_center[grasp_idx]
        x_ft_diff = grasp_width_multiple * cand_diff[grasp_idx]
        x_lft_t = x_ft_center + x_ft_diff
        x_rft_t = x_ft_center - x_ft_diff

        ft_x_thumb_target = x_lft_t
        diff = x_rft_t - x_lft_t
        diff_u = diff / (np.linalg.norm(diff) + 1e-9)
        unit_vec = np.cross(self._grasp_reach_dir_tf, diff_u)
        un = np.linalg.norm(unit_vec)
        if un < 1e-8:
            unit_vec = np.cross(np.array([0.0, 0.0, 1.0]), diff_u)
            un = np.linalg.norm(unit_vec) + 1e-9
        unit_vec = unit_vec / un

        cand1_i = x_rft_t + unit_vec * 0.02
        cand1_m = x_rft_t - unit_vec * 0.02
        cand2_i = x_rft_t - unit_vec * 0.02
        cand2_m = x_rft_t + unit_vec * 0.02
        cur = (ft_x_index - ft_x_middle) / (
            np.linalg.norm(ft_x_index - ft_x_middle) + 1e-9
        )
        d1 = (cand1_i - cand1_m) / (np.linalg.norm(cand1_i - cand1_m) + 1e-9)
        d2 = (cand2_i - cand2_m) / (np.linalg.norm(cand2_i - cand2_m) + 1e-9)
        e1 = np.arccos(np.clip(d1.dot(cur), -1.0, 1.0))
        e2 = np.arccos(np.clip(d2.dot(cur), -1.0, 1.0))
        prefer_1 = e1 < e2

        if update_selection:
            if self._index_middle_assignment is None:
                self._index_middle_assignment = 0 if prefer_1 else 1
            # Sticky: only flip assignment if the other is clearly better.
            elif prefer_1 and self._index_middle_assignment == 1 and (e2 - e1) > 0.25:
                self._index_middle_assignment = 0
            elif (
                (not prefer_1)
                and self._index_middle_assignment == 0
                and (e1 - e2) > 0.25
            ):
                self._index_middle_assignment = 1
            use_1 = self._index_middle_assignment == 0
        else:
            use_1 = (
                prefer_1
                if self._index_middle_assignment is None
                else self._index_middle_assignment == 0
            )

        if use_1:
            return ft_x_thumb_target, cand1_i, cand1_m
        return ft_x_thumb_target, cand2_i, cand2_m

    def _tip_positions(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros((3, 3), dtype=np.float64)
        for i, name in enumerate(("thumb_tip", "index_tip", "middle_tip")):
            out[i] = get_fk_link_pose(q, self._pin_model, self._pin_data_plan, name)[
                :3, 3
            ]
        return out

    def _enter_reaching(self) -> None:
        self._state = "reaching"
        self._reset_closing_flags()
        # Re-select grasp once when (re)entering reach; then keep it sticky.
        self._grasp_selection_locked = False
        self._index_middle_assignment = None

    def _fsm(self) -> None:
        if self._state == "reaching":
            if self._error_reaching < DIST_THR_REACHING2CLOSING_GRIPPER:
                print(f"[3FGrasp] reach→close (e={self._error_reaching:.3f})")
                self._enter_closing()
        elif self._state == "closing_gripper":
            # Lost grasp (incl. after lift hold) → re-reach.
            if self._error_grasp > DIST_THR_ANY2REACHING:
                print(f"[3FGrasp] close→reach (lost grasp, e={self._error_grasp:.3f})")
                self._enter_reaching()
                return

            if self._closing_t0 is None:
                self._closing_t0 = time.time()
            elapsed = time.time() - self._closing_t0
            if (not self._lift_active) and elapsed >= self._lift_delay_s:
                self._lift_active = True
                self._lift_done = False
                print(
                    f"[3FGrasp] squeeze done ({elapsed:.1f}s) → arm joints to default_q"
                )
            elif self._lift_active and (not self._lift_done):
                e_arm = float(
                    np.linalg.norm(
                        self._q[: self._num_arm] - self._q_nominal[: self._num_arm]
                    )
                )
                if e_arm < self._lift_arm_eps:
                    self._lift_done = True
                    print(f"[3FGrasp] arm at default (e={e_arm:.3f}); holding squeeze")

    def _compute_fingertip_linear_velocity_des(
        self,
        ft_x_thumb,
        ft_x_index,
        ft_x_middle,
        ft_x_thumb_target,
        ft_x_index_target,
        ft_x_middle_target,
        obj=None,
        other_obstacles=None,
    ):
        other_obstacles = list(other_obstacles or [])

        def dist_func(x):
            if obj is None:
                return np.full(x.shape[0], 1.0)
            dist, _grad = obj.get_dist_and_grad(x)
            return dist

        def dist_grad_func(x):
            if obj is None:
                n = x.shape[0]
                return np.full(n, 1.0), np.zeros((n, 3))
            return obj.get_dist_and_grad(x)

        ft_x_init = np.concatenate([ft_x_thumb, ft_x_index, ft_x_middle], axis=0)
        ft_x_goal = np.concatenate(
            [ft_x_thumb_target, ft_x_index_target, ft_x_middle_target], axis=0
        )
        ft_traj = heuristic_path_initialization_general(
            ft_x_init,
            ft_x_goal,
            x_dim=3,
            num_samples=self._num_lvf_points,
            traj_length=self._num_lvf_points,
            length=self._heuristic_via_length,
            constraints_margin=self._col_margin_lvf,
            dist_func=dist_func,
        )
        ft_traj = np.asarray(ft_traj, dtype=np.float64)

        # SQP polish (ICRA TO reaching): smooth + clearance linearizations.
        if self._num_path_sqp_iterations > 0 and obj is not None:
            other_fns = [
                (lambda o: (lambda x: o.get_dist_and_grad(x)))(obs)
                for obs in other_obstacles
                if hasattr(obs, "get_dist_and_grad")
            ]
            ft_traj = smooth_multi_finger_paths_sqp(
                ft_traj,
                dist_grad_func,
                num_iterations=self._num_path_sqp_iterations,
                constraints_margin=self._col_margin_lvf,
                other_dist_grad_funcs=other_fns,
                other_constraints_margin=self._col_margin_lvf,
                savgol_window=self._path_sqp_savgol_window,
            )

        self._ft_paths = ft_traj

        plan_dt = max(float(self._plan_dt), 1e-3)
        outs = []
        tips = (ft_x_thumb, ft_x_index, ft_x_middle)
        tgts = (ft_x_thumb_target, ft_x_index_target, ft_x_middle_target)
        for i in range(3):
            traj = self._ft_paths[i]
            direction = traj[1] - traj[0]
            n = float(np.linalg.norm(direction))
            if n < 1e-9:
                direction = tgts[i] - tips[i]
                n = float(np.linalg.norm(direction))
            if n < 1e-9:
                outs.append(np.zeros(3))
                continue
            direction = direction / n
            err = float(np.linalg.norm(tips[i] - tgts[i]))
            speed = speed_const_then_linear_clamped(
                err, self._lvf_v, self._lvf_eps, plan_dt
            )
            outs.append(speed * direction)
        return outs[0], outs[1], outs[2]

    def _get_col_links(self, q: np.ndarray) -> dict:
        poses = get_fk_link_poses(
            q, self._pin_model, self._pin_data_plan, self._col_link_names
        )
        jacs = get_link_Jacobians(
            q, self._pin_model, self._pin_data_plan, self._col_link_names
        )
        out = {}
        for i, link_name in enumerate(self._col_link_names):
            if link_name not in self._fcl_robot_col_geoms:
                continue
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
            if obs_name not in self._fcl_static_col_geoms:
                continue
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
            if link_name not in dict_col_links or obs_name not in dict_col_obstacles:
                continue
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

    def _collision_escape_lb(self, dist: float, margin: float) -> float:
        """Lower bound on outward speed; clipped to avoid infeasible QPs."""
        req = (-float(dist) + float(margin)) / max(self._plan_horizon, 1e-6)
        return float(min(req, self._ikqp_max_escape_rate))

    def _add_collision2env_constraints(
        self, list_A, list_A_lb, list_A_ub, collision_data, dist_margin=0.01
    ):
        for link_name, obs_name in self._col_pairs:
            if obs_name not in collision_data.get(link_name, {}):
                continue
            col = collision_data[link_name][obs_name]
            n = col["nearest_p_link"] - col["nearest_p_obs"]
            d = float(np.clip(col["dist"], 1e-6, None))
            list_A.append((n.T @ col["point_Jac"] / d).reshape(1, -1).copy())
            list_A_lb.append(
                np.array([self._collision_escape_lb(col["dist"], dist_margin)])
            )
            list_A_ub.append(np.array([1000.0 / self._plan_horizon]))
        return list_A, list_A_lb, list_A_ub

    def _add_collision2obstacles_constraints(
        self,
        list_A,
        list_A_lb,
        list_A_ub,
        list_of_x,
        list_of_Jx,
        list_objects,
        dist_margin=0.01,
    ):
        x_query = np.vstack([x.reshape(1, 3) for x in list_of_x])
        for se in list_objects:
            dist, grad_dist = se.get_dist_and_grad(x_query)
            n_pts = len(dist)
            for i, (d, gd, J_point) in enumerate(zip(dist, grad_dist, list_of_Jx)):
                # brl: last point uses 2x margin (palm); our list has no palm —
                # keep x1 for all phalanx / fingertip queries.
                dist_margin_multiple = 2.0 if i == n_pts - 1 else 1.0
                list_A.append((gd.reshape(1, 3) @ J_point).copy())
                list_A_lb.append(
                    np.array(
                        [
                            self._collision_escape_lb(
                                d, dist_margin * dist_margin_multiple
                            )
                        ]
                    )
                )
                list_A_ub.append(np.array([1000.0 / self._plan_horizon]))
        return list_A, list_A_lb, list_A_ub

    def _solve_IKQP(
        self, q, task_space_plans, list_A=None, list_A_lb=None, list_A_ub=None
    ):
        """Weighted task-space IK QP (BRL failure policy).

        On infeasible / unsolved OSQP: return ``qd = 0`` for **this plan only**.
        Do not latch a permanent fault, and never command OSQP's infeasibility
        certificate in ``res.x``.
        """
        vlim = self._joint_velocity_limit

        # Joint velocity / limit box.
        l_j = np.maximum(-vlim, (self._q_lower - q) / self._plan_horizon)
        u_j = np.minimum(vlim, (self._q_upper - q) / self._plan_horizon)
        # If q is outside soft limits, box can be empty → repair to zeros.
        bad = l_j > u_j
        if np.any(bad):
            l_j[bad] = 0.0
            u_j[bad] = 0.0

        if list_A and list_A_lb is not None and list_A_ub is not None:
            A_extra = np.vstack(list_A)
            l_extra = np.asarray(np.hstack(list_A_lb), dtype=np.float64).reshape(-1)
            u_extra = np.asarray(np.hstack(list_A_ub), dtype=np.float64).reshape(-1)
            # Drop / repair individually inconsistent rows.
            bad_row = l_extra > u_extra
            if np.any(bad_row):
                l_extra[bad_row] = 0.0
                u_extra[bad_row] = 0.0
            A = sparse.csc_matrix(np.vstack([np.eye(self._n_x), A_extra]))
            l = np.hstack([l_j, l_extra])
            u = np.hstack([u_j, u_extra])
        else:
            A = sparse.csc_matrix(np.eye(self._n_x))
            l = l_j
            u = u_j

        # Cost: Σ w ||J qd - xd||² + qdᵀ diag(reg) qd + rate ||qd - qd_prev||²
        P_np = np.diag(self._ikqp_qd_reg.copy())
        g_np = np.zeros(self._n_x)
        if self._ikqp_reg_qd_rate > 0.0:
            P_np += np.eye(self._n_x) * self._ikqp_reg_qd_rate
            g_np -= self._ikqp_reg_qd_rate * self._qd_des
        for w, J, xd_des in task_space_plans:
            if w <= 0.0:
                continue
            P_np += J.T @ J * w
            g_np += -xd_des @ J * w

        # BRL: if u < l → no solution → qd = 0 (this plan only).
        if np.any(u < l - 1e-12):
            return self._ikqp_fail_zeros("u < l (inconsistent bounds)")

        try:
            m = osqp.OSQP()
            m.setup(
                P=sparse.csc_matrix(P_np),
                q=g_np,
                A=A,
                l=l,
                u=u,
                verbose=False,
                warm_starting=True,
            )
            # Warm-start from previous command (clipped).
            x0 = np.clip(self._qd_des, -vlim, vlim)
            m.warm_start(x=x0, y=np.zeros(A.shape[0]))
            res = m.solve()
        except Exception as e:
            return self._ikqp_fail_zeros(f"exception: {e}")

        status_val = int(getattr(res.info, "status_val", -1))
        solved = status_val == int(osqp.constant("OSQP_SOLVED"))
        # BRL: only accept OSQP_SOLVED; otherwise qd = 0 this plan.
        # Never use res.x on failure (infeasibility certificate is huge).
        if not solved:
            return self._ikqp_fail_zeros(f"status={res.info.status}")

        qd = np.asarray(res.x, dtype=np.float64).reshape(-1)
        if qd.shape[0] != self._n_x or not np.all(np.isfinite(qd)):
            return self._ikqp_fail_zeros("non-finite solution")

        # Reject absurd magnitudes (should not happen if solved cleanly).
        if np.any(np.abs(qd) > 1.05 * vlim + 1e-6):
            return self._ikqp_fail_zeros("|qd| exceeds limits")
        return np.clip(qd, -vlim, vlim)

    def _plan_reaching(self):
        ######################
        # Compute FT states  #
        ######################
        q = self._q.copy()

        ##############################################
        # Compute target threefinger linear velocity #
        ##############################################
        obj_to_grasp = self._target_obj if self._have_obj else None
        list_non_target_obstacles = []
        ft_xd_thumb_des, ft_xd_index_des, ft_xd_middle_des = (
            self._compute_fingertip_linear_velocity_des(
                self._ft_x_thumb,
                self._ft_x_index,
                self._ft_x_middle,
                self._ft_x_thumb_target,
                self._ft_x_index_target,
                self._ft_x_middle_target,
                obj=obj_to_grasp,
                other_obstacles=list_non_target_obstacles,
            )
        )

        ###############################################
        # Compute target threefinger angular velocity #
        ###############################################
        omega_thumb_des = np.zeros(3)
        omega_index_des = OMEGA_INDEX_GAIN * np.cross(
            self._ft_n_index, self._ft_n_index_target
        )
        omega_middle_des = OMEGA_MIDDLE_GAIN * np.cross(
            self._ft_n_middle, self._ft_n_middle_target
        )

        ##########################
        # gripper velocity field #
        ##########################
        q_gripper = q[7:]
        qd_gripper_des = goto_x_des_const_then_linear_clamped(
            q_gripper,
            self.q_nominal_gripper_pose,
            GRIPPER_NOMINAL_V,
            GRIPPER_NOMINAL_EPS,
            0.05,  # 20 Hz
        )

        ##############################
        # position-dependent weights #
        ##############################
        w_reaching = weight_func_tanh(self._error_reaching, 0.1, 0.01)
        w_reaching2 = weight_func_tanh(self._error_reaching, 0.03, 0.01)
        w_index = weight_func_tanh(
            self._error_index_angle, np.pi / 180 * 30.0, np.pi / 180 * 5.0
        )
        w_middle = weight_func_tanh(
            self._error_middle_angle, np.pi / 180 * 30.0, np.pi / 180 * 5.0
        )

        w_ft_x_thumb = 1.0
        w_ft_x_index = 1.0
        w_ft_x_middle = 1.0
        w_R_thumb = 0.0
        w_R_index = w_index * (1 - w_reaching2)
        w_R_middle = w_middle * (1 - w_reaching2)
        w_q_gripper = 1.0 * w_reaching

        task_space_plans = [
            (w_ft_x_thumb, self._J_ft_x_thumb, ft_xd_thumb_des),
            (w_ft_x_index, self._J_ft_x_index, ft_xd_index_des),
            (w_ft_x_middle, self._J_ft_x_middle, ft_xd_middle_des),
            (w_R_thumb, self._J_R["thumb_tip"], omega_thumb_des),
            (w_R_index, self._J_R["index_tip"], omega_index_des),
            (w_R_middle, self._J_R["middle_tip"], omega_middle_des),
            (w_q_gripper, self._J_q_gripper, qd_gripper_des),
        ]

        ###############
        # Constraints #
        ###############
        list_A: list = []
        list_A_lb: list = []
        list_A_ub: list = []

        dict_col_links = self._get_col_links(q)
        dict_col_obstacles = self._get_col_obstacles()
        collision_data = self._get_col_data(dict_col_links, dict_col_obstacles)

        list_A, list_A_lb, list_A_ub = self._add_collision2env_constraints(
            list_A,
            list_A_lb,
            list_A_ub,
            collision_data,
            dist_margin=self._env_dist_margin,
        )

        #############################################################
        # collision between robot and object to grasp and obstacles #
        #############################################################
        list_of_x = [
            self._poses["finger_r_link_1_thumb2"][:3, 3],
            self._poses["finger_r_link_1_thumb3"][:3, 3],
            self._poses["finger_r_link_1_thumb4"][:3, 3],
            self._ft_x_thumb,
            self._poses["finger_r_link_2_index2"][:3, 3],
            self._poses["finger_r_link_2_index3"][:3, 3],
            self._poses["finger_r_link_2_index4"][:3, 3],
            self._ft_x_index,
            self._poses["finger_r_link_3_middle2"][:3, 3],
            self._poses["finger_r_link_3_middle3"][:3, 3],
            self._poses["finger_r_link_3_middle4"][:3, 3],
            self._ft_x_middle,
            self._poses["finger_r_link_4_ring2"][:3, 3],
            self._poses["finger_r_link_4_ring3"][:3, 3],
            self._poses["finger_r_link_4_ring4"][:3, 3],
            self._poses["finger_r_link_5_little2"][:3, 3],
            self._poses["finger_r_link_5_little3"][:3, 3],
            self._poses["finger_r_link_5_little4"][:3, 3],
        ]
        list_of_Jx = [
            self._J_x["finger_r_link_1_thumb2"],
            self._J_x["finger_r_link_1_thumb3"],
            self._J_x["finger_r_link_1_thumb4"],
            self._J_ft_x_thumb,
            self._J_x["finger_r_link_2_index2"],
            self._J_x["finger_r_link_2_index3"],
            self._J_x["finger_r_link_2_index4"],
            self._J_ft_x_index,
            self._J_x["finger_r_link_3_middle2"],
            self._J_x["finger_r_link_3_middle3"],
            self._J_x["finger_r_link_3_middle4"],
            self._J_ft_x_middle,
            self._J_x["finger_r_link_4_ring2"],
            self._J_x["finger_r_link_4_ring3"],
            self._J_x["finger_r_link_4_ring4"],
            self._J_x["finger_r_link_5_little2"],
            self._J_x["finger_r_link_5_little3"],
            self._J_x["finger_r_link_5_little4"],
        ]

        if obj_to_grasp is not None:
            list_A, list_A_lb, list_A_ub = self._add_collision2obstacles_constraints(
                list_A,
                list_A_lb,
                list_A_ub,
                list_of_x,
                list_of_Jx,
                [obj_to_grasp],
                dist_margin=self._col_margin_obj_qp,
            )

        ########
        # IKQP #
        ########
        _plan_qd_des = self._solve_IKQP(
            q,
            task_space_plans,
            list_A=list_A,
            list_A_lb=list_A_lb,
            list_A_ub=list_A_ub,
        )
        return _plan_qd_des, np.zeros_like(self._q)

    def _plan_closing(self):
        """Squeeze grasp; after delay, send arm joints to default_q then hold."""
        q = self._q.copy()
        plan_dt = max(float(self._plan_dt), 1e-3)
        qd = np.zeros(self._n, dtype=np.float64)

        if self._lift_active:
            if not self._lift_done:
                qd[: self._num_arm] = goto_x_des_const_then_linear_clamped(
                    q[: self._num_arm],
                    self._q_nominal[: self._num_arm],
                    self._lift_speed,
                    self._lift_arm_eps,
                    plan_dt,
                )
        else:
            tips = np.vstack([self._ft_x_thumb, self._ft_x_index, self._ft_x_middle])
            tgts = np.vstack(
                [
                    self._ft_x_thumb_target,
                    self._ft_x_index_target,
                    self._ft_x_middle_target,
                ]
            )
            v_tips = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                v_tips[i] = goto_x_des_const_then_linear_clamped(
                    tips[i], tgts[i], 0.25, 0.02, plan_dt
                )
            task_space_plans = [
                (1.0, self._J_ft_x_thumb, v_tips[0]),
                (1.0, self._J_ft_x_index, v_tips[1]),
                (1.0, self._J_ft_x_middle, v_tips[2]),
            ]
            qd = self._solve_IKQP(q, task_space_plans)

        q_des = self._q.copy()
        q_des[self._num_arm :] = self._q_hand_close
        tau = 0.8 * self._kp * (q_des - self._q)
        tau[: self._num_arm] = 0.0
        return qd, tau

    def _plan_loop(self) -> None:
        last = 0.0
        while not self._stop_plan.is_set() and not self._finished:
            now = time.time()
            if now - last < self._plan_dt:
                time.sleep(0.0005)
                continue
            last = now
            if self._state in ("grav_comp", "default"):
                continue
            if not self._have_obj:
                continue
            try:
                self._compute_variables()
                self._fsm()
                if self._state == "reaching":
                    qd, tau = self._plan_reaching()
                elif self._state == "closing_gripper":
                    qd, tau = self._plan_closing()
                else:
                    continue
                try:
                    while True:
                        self._plan_results.get_nowait()
                except Empty:
                    pass
                self._plan_results.put((qd, tau))
            except Exception as e:
                print(f"[3FGrasp] plan error: {e}")

    def _update(self) -> None:
        q = self._q.copy()
        tau_ff = np.zeros(self._n, dtype=np.float64)
        if self._do_grav_comp:
            tau_ff = pin.computeGeneralizedGravity(self._pin_model, self._pin_data, q)

        if self._state == "grav_comp":
            q_des = q
            qd_des = np.zeros(self._n)
            kp = np.zeros(self._n)
            kd = np.zeros(self._n)
        elif self._state == "default":
            qd_des = goto_x_des_const_then_linear_clamped(
                q, self._q_nominal, DEFAULT_V, DEFAULT_EPS, self._ctrl_dt
            )
            q_des = q
            kp = np.zeros(self._n)
            kd = self._vel_kd
        else:
            if not self._plan_results.empty():
                self._qd_des, self._tau_ff_des = self._plan_results.get()
            qd_des = self._qd_des.copy()
            q_des = q
            kp = np.zeros(self._n)
            kd = self._vel_kd
            tau_ff = tau_ff + self._tau_ff_des
            if self._do_friction_comp:
                phi = 0.03
                tau_ff = tau_ff + self._Fjc * np.clip(qd_des / phi, -1.0, 1.0)

        cmd = JointCtrlData(num_joints=self._n)
        cmd.set_data(self._cur_time, q_des, qd_des, tau_ff, kp, kd)
        self.ctrl_pub_que.put(cmd)
        self._publish_grasp_viz()

    def _put_named_vec(self, channel: str, name_list, vec_list) -> None:
        if self.pub_que_dict is None or not channel:
            return
        que = self.pub_que_dict.get(channel)
        if que is None:
            return
        vecs = np.asarray(vec_list, dtype=np.float64)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        msg = NamedVecListData(
            num_vecs=len(name_list), vec_dim=int(vecs.shape[1]), name=channel
        )
        msg.set_data(float(self._cur_time), name_list, vecs)
        try:
            while not que.empty():
                que.get_nowait()
        except Exception:
            pass
        try:
            que.put_nowait(msg)
        except Exception:
            pass

    def _publish_contact_arrows(
        self,
        points_world: np.ndarray,
        normals_world: np.ndarray,
        forces_world: np.ndarray | None = None,
        *,
        clear_forces: bool = False,
    ) -> None:
        """Publish contact arrows as NamedVecList rows ``[p|v]`` (6-D, world).

        - Cyan normals are always updated from ``points_world`` / ``normals_world``.
        - Magenta forces update only when ``forces_world`` is provided; pass
          ``clear_forces=True`` (or empty points) to wipe stale force arrows.
        """
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        nrms = np.asarray(normals_world, dtype=np.float64).reshape(-1, 3)
        P = int(pts.shape[0])

        if self._contact_normal_arrows_channel:
            if P == 0:
                self._put_named_vec(
                    self._contact_normal_arrows_channel,
                    ["empty"],
                    np.zeros((1, 6)),
                )
            else:
                n = nrms / np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-8)
                self._put_named_vec(
                    self._contact_normal_arrows_channel,
                    [str(i) for i in range(P)],
                    np.concatenate([pts, n], axis=1),
                )

        if self._contact_force_arrows_channel:
            if P == 0 or clear_forces:
                self._put_named_vec(
                    self._contact_force_arrows_channel,
                    ["empty"],
                    np.zeros((1, 6)),
                )
            elif forces_world is not None:
                f = np.asarray(forces_world, dtype=np.float64).reshape(-1, 3)
                self._put_named_vec(
                    self._contact_force_arrows_channel,
                    [str(i) for i in range(P)],
                    np.concatenate([pts, f], axis=1),
                )

    def _publish_grasp_viz(self) -> None:
        self._viz_tick += 1
        if self._viz_tick % max(1, self._viz_stride) != 0:
            return
        if self._have_obj:
            gps = self._grasp_points_world()
            names, vecs = [], []
            for i, pair in enumerate(gps):
                names.extend([f"c{i}_a", f"c{i}_b"])
                vecs.extend([pair[0], pair[1]])
            self._put_named_vec(self._grasp_candidates_channel, names, vecs)

        tips = self._tip_positions(self._q)
        ft_names = [
            "thumb_cur",
            "index_cur",
            "middle_cur",
            "thumb_tgt",
            "index_tgt",
            "middle_tgt",
        ]
        # Do not publish origin (0,0,0) targets before the first plan update —
        # that looks like targets jumping between the mesh and "default".
        if self._have_obj and self._ft_targets_valid:
            tgts = self._ft_targets.copy()
        else:
            tgts = tips.copy()
        ft_vecs = np.vstack([tips, tgts])
        self._put_named_vec(self._fingertip_viz_channel, ft_names, ft_vecs)

        # Heuristic fingertip LVF paths (waypoints), updated during reaching.
        if self._ft_path_viz_channel and self._state == "reaching":
            paths = np.asarray(self._ft_paths, dtype=np.float64)
            if paths.ndim == 3 and paths.shape[0] == 3 and paths.shape[2] == 3:
                names, vecs = [], []
                for fi, fname in enumerate(("thumb", "index", "middle")):
                    for ti in range(paths.shape[1]):
                        names.append(f"{fname}_{ti}")
                        vecs.append(paths[fi, ti])
                self._put_named_vec(self._ft_path_viz_channel, names, vecs)

        # Clear contact overlays outside closing so arrows don't linger.
        if self._state != "closing_gripper":
            self._publish_contact_arrows(
                np.zeros((0, 3)), np.zeros((0, 3)), clear_forces=True
            )


def _quat_wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm([w, x, y, z]))
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
