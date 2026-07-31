"""Three-finger reactive grasp with real-time force-closure closing.

Reaching is identical to ``ThreeFingerReactiveGrasping`` (fingertip LVF →
orientation / gripper VFs → collision-aware IKQP).

After reach→close, squeeze and lift replace the hand PD close torque with the
real-time contact / force-closure stack from ``MjEvalGraspControl``:

  UDF contact detection → attraction → fingertip approach → force-closure QP → τ

Arm lift (joint VF → ``default_q``) is unchanged. Hand force regulation runs
in the control loop (``ctrl_freq``), not the 50 Hz planner.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh
from omegaconf import DictConfig

from controller.MjEvalGraspControl import HAND_LINK_NAMES, N_HAND_JOINTS
from controller.ThreeFingerReactiveGrasping import ThreeFingerReactiveGrasping
from data_type.basic_types.JointCtrlData import JointCtrlData
from utils.grasping.distance_field import build_distance_field, detect_contacts
from utils.grasping.force_gen import compute_contact_frame, generate_contact_forces
from utils.grasping.link_surface_sampler import load_link_surface_samples
from utils.lie.kinematics import get_point_Jacobian
from utils.pinocchio.getter import get_fk_link_poses, get_link_Jacobians
from utils.velocity_fields.grasping import goto_x_des_const_then_linear_clamped

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_PROJECT_ROOT, p))


class ThreeFingerReactiveForceClosureGrasping(ThreeFingerReactiveGrasping):
    """Reach with reactive hierarchy; close/lift with RT force-closure τ."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        gen_cfg = dict(config.get("grasp_squeeze_params_generator_config", {}) or {})
        self._gen_config = gen_cfg
        self._obj_mesh_path_force = _resolve_path(
            str(
                config.get(
                    "obj_mesh_path",
                    f"assets/object_mesh/{self._obj_name}/textured_mesh.obj",
                )
            )
        )
        self._force_ready = False
        self._prev_tau_qp = np.zeros(N_HAND_JOINTS, dtype=np.float64)
        self._approach_qdot_des_avg = np.zeros(N_HAND_JOINTS, dtype=np.float64)
        self._force_poses: dict = {}
        self._force_Jacobians: dict = {}
        self._contact_points_world_vis = np.zeros((0, 3))
        self._f_world_vis = np.zeros((0, 3))
        self._rt_contact_normals_world_vis = np.zeros((0, 3))
        self._rt_qp_contact_pts_world_vis = np.zeros((0, 3))
        self._cad_in_world_frame = np.eye(4)
        self._world2cad = np.eye(4)
        self._obj_center_in_cad_frame = np.zeros(3)
        self._k_for_contact_force_control = float(
            gen_cfg.get("k_for_contact_force_control", 10.0)
        )
        # Tip attractors call 27-DoF IKQP into the object and fight force τ —
        # leave OFF; pre-contact motion comes from UDF fingertip approach.
        self._closing_tip_attractors = bool(config.get("closing_tip_attractors", False))
        # Force QP is expensive; run at this rate and hold τ between solves so
        # the 1 kHz velocity loop stays real-time.
        self._force_ctrl_freq = float(config.get("force_ctrl_freq", 50.0))
        self._force_ctrl_dt = 1.0 / max(self._force_ctrl_freq, 1.0)
        self._last_force_solve_t = -1e9
        self._tau_hand_hold = np.zeros(N_HAND_JOINTS, dtype=np.float64)
        self._force_qp_fail_count = 0
        self._force_qp_fail_log_t = -1e9
        # Gentle PD toward q_hand_close when force QP has no feasible contacts.
        self._closing_pd_scale = float(config.get("closing_pd_scale", 0.25))
        # Dedicated Pinocchio Data for force FK (control thread only).
        self._pin_data_force = self._pin_model.createData()

        print(
            "[3FForceClosure] Reaching = ThreeFingerReactiveGrasping; "
            "closing/lift = RT force-closure (MjEvalGraspControl)."
        )

    def initialize(self) -> None:
        """Wait for FCL, build force module, *then* start the plan thread.

        Starting the planner before UDF/surface init (via ``super().initialize``)
        races the 50 Hz plan loop against a multi-second GPU build and can make
        early reaching look different from tutorial 10.
        """
        self._fcl_robot_col_geoms = self._wait_for_col_geoms(
            self.intr_sub_que_dict[self._robot_col_info_channel], "robot"
        )
        self._fcl_static_col_geoms = self._wait_for_col_geoms(
            self.extr_sub_que_dict[self._static_col_info_channel], "static"
        )
        self._init_force_module()
        self._force_ready = True
        self._plan_thread.start()
        print(
            f"[3FForceClosure] Initialized "
            f"(distance_field={self._distance_field_backend}; plan thread started)."
        )

    def _init_force_module(self) -> None:
        """Build UDF + hand surface samples (hand-only URDF for collision meshes)."""
        gen = self._gen_config
        urdf_path = _resolve_path(
            gen.get(
                "urdf_path",
                "assets/scene/robotis_5f/robotis_5f.urdf",
            )
        )
        mesh_path_cfg = gen.get("mesh_path", "assets/scene/robotis_5f")
        if isinstance(mesh_path_cfg, dict):
            mesh_paths = [_resolve_path(p) for p in mesh_path_cfg.values()]
        else:
            mesh_paths = [_resolve_path(str(mesh_path_cfg))]
        hand_mesh_dir = mesh_paths[0]

        obj_mesh = Path(self._obj_mesh_path_force)
        if not obj_mesh.exists():
            raise FileNotFoundError(f"Object mesh missing: {obj_mesh}")
        mesh = trimesh.load(str(obj_mesh))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        self._force_mesh = mesh
        self._obj_center_in_cad_frame = np.asarray(mesh.centroid, dtype=np.float64)

        udf_res = int(gen.get("udf_resolution", 128))
        udf_pad = float(gen.get("udf_padding", 0.02))
        udf_device = str(gen.get("udf_device", "cuda:0"))
        udf_backend = str(gen.get("distance_field_backend", "auto"))
        self._udf_grid, self._distance_field_backend = build_distance_field(
            mesh,
            resolution=udf_res,
            padding=udf_pad,
            device=udf_device,
            backend=udf_backend,
        )

        density = gen.get("surface_density_per_cm2", 2.0)
        finger_links = [l for l in HAND_LINK_NAMES if l != "palm"]
        for name in HAND_LINK_NAMES:
            if not self._pin_model.existFrame(name):
                raise ValueError(
                    f"Force-closure link frame {name!r} missing in arm+hand URDF"
                )
        finger_samples, finger_normals = load_link_surface_samples(
            urdf_path,
            hand_mesh_dir,
            finger_links,
            density_per_cm2=density,
            return_normals=True,
        )
        palm_samples, palm_normals = load_link_surface_samples(
            urdf_path,
            hand_mesh_dir,
            ["palm"],
            density_per_cm2=density,
            keep_normal_dirs={"palm": [1, 0, 0]},
            normal_dot_threshold=0.5,
            keep_position_filters={"palm": ([1, 0, 0], 0.02)},
            return_normals=True,
        )
        self._hand_surface_samples_per_link = finger_samples + palm_samples
        self._hand_surface_normals_per_link = finger_normals + palm_normals
        self._hand_link_names = finger_links + ["palm"]
        self._palm_link_idx = len(finger_links)

    # ------------------------------------------------------------------
    # Closing plan: arm lift only; hand τ comes from control-loop force stack
    # ------------------------------------------------------------------

    def _plan_closing(self):
        """Arm lift VF after delay; hand motion via force τ (not tip IKQP).

        Tip-attractor IKQP digs into the object, trips / fights the force-closure
        OSQP, and is the usual source of “goes crazy” on close. Keep hand
        ``qd = 0`` here; UDF approach + QP provide squeeze.
        """
        q = self._q.copy()
        plan_dt = max(float(self._plan_dt), 1e-3)
        qd = np.zeros(self._n, dtype=np.float64)

        if self._lift_active and (not self._lift_done):
            qd[: self._num_arm] = goto_x_des_const_then_linear_clamped(
                q[: self._num_arm],
                self._q_nominal[: self._num_arm],
                self._lift_speed,
                self._lift_arm_eps,
                plan_dt,
            )
        elif self._closing_tip_attractors and (not self._lift_active):
            # Opt-in only; hand columns still zeroed in ``_update``.
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
                    tips[i], tgts[i], 0.15, 0.02, plan_dt
                )
            task_space_plans = [
                (1.0, self._J_ft_x_thumb, v_tips[0]),
                (1.0, self._J_ft_x_index, v_tips[1]),
                (1.0, self._J_ft_x_middle, v_tips[2]),
            ]
            qd = self._solve_IKQP(q, task_space_plans)
            # Never command arm from tip attractors during squeeze.
            qd[: self._num_arm] = 0.0

        return qd, np.zeros(self._n, dtype=np.float64)

    def _enter_closing(self) -> None:
        super()._enter_closing()
        # Clear any reaching IKQP latch so closing force control can run.
        self._clear_ikqp_fault()
        self._prev_tau_qp[:] = 0.0
        self._tau_hand_hold[:] = 0.0
        self._approach_qdot_des_avg[:] = 0.0
        self._force_qp_fail_count = 0
        self._last_force_solve_t = -1e9
        print(
            "[3FForceClosure] closing: RT force-closure "
            f"(tip_attractors={self._closing_tip_attractors}, "
            f"force_ctrl_freq={self._force_ctrl_freq:.0f} Hz)"
        )

    # ------------------------------------------------------------------
    # Control loop: inject hand force τ during closing / lift
    # ------------------------------------------------------------------

    def _update(self) -> None:
        # Reaching / grav / default: identical to parent (no duplicated control path).
        if (
            self._state != "closing_gripper"
            or not self._force_ready
            or not self._have_obj
        ):
            super()._update()
            return

        q = self._q.copy()
        tau_ff = np.zeros(self._n, dtype=np.float64)
        if self._do_grav_comp:
            tau_ff = pin.computeGeneralizedGravity(self._pin_model, self._pin_data, q)

        if not self._plan_results.empty():
            self._qd_des, self._tau_ff_des = self._plan_results.get()

        qd_des = self._qd_des.copy()
        # Hand squeeze is torque-only; zero hand velocity commands so damping
        # does not fight / amplify force-closure τ.
        qd_des[self._num_arm :] = 0.0
        q_des = q
        kp = np.zeros(self._n)
        kd = self._vel_kd.copy()
        # Soft hand damping only (arm keeps tracking lift qd).
        kd[self._num_arm :] = np.minimum(kd[self._num_arm :], 0.2)

        tau_ff = tau_ff + self._tau_ff_des
        tau_hand = self._compute_rt_force_torque_rate_limited()
        tau_ff[self._num_arm :] = tau_ff[self._num_arm :] + tau_hand
        self._publish_force_contact_arrows()

        cmd = JointCtrlData(num_joints=self._n)
        cmd.set_data(self._cur_time, q_des, qd_des, tau_ff, kp, kd)
        self.ctrl_pub_que.put(cmd)
        self._publish_grasp_viz()

    def _hand_torque_limit(self) -> np.ndarray:
        lim = np.asarray(
            self._gen_config.get("torque_limit", [0.2] * N_HAND_JOINTS),
            dtype=np.float64,
        ).reshape(-1)
        if lim.shape[0] != N_HAND_JOINTS:
            lim = np.full(N_HAND_JOINTS, 0.2, dtype=np.float64)
        return lim

    def _clip_hand_tau(self, tau: np.ndarray) -> np.ndarray:
        lim = self._hand_torque_limit()
        tau = np.asarray(tau, dtype=np.float64).reshape(N_HAND_JOINTS)
        if not np.all(np.isfinite(tau)):
            return np.zeros(N_HAND_JOINTS, dtype=np.float64)
        return np.clip(tau, -lim, lim)

    def _closing_pd_torque(self) -> np.ndarray:
        """Safe fallback toward ``q_hand_close`` when force QP is infeasible."""
        q_h = self._q[self._num_arm :]
        q_des = self._q_hand_close
        kp = self._kp[self._num_arm :]
        tau = self._closing_pd_scale * kp * (q_des - q_h)
        return self._clip_hand_tau(tau)

    def _compute_rt_force_torque_rate_limited(self) -> np.ndarray:
        now = float(self._cur_time)
        if (now - self._last_force_solve_t) < self._force_ctrl_dt:
            return self._tau_hand_hold
        self._last_force_solve_t = now
        tau = self._compute_rt_force_torque()
        self._tau_hand_hold = self._clip_hand_tau(tau)
        return self._tau_hand_hold

    def _log_force_qp_fail(self, msg: str) -> None:
        self._force_qp_fail_count += 1
        now = time.time()
        if now - self._force_qp_fail_log_t < 1.0:
            return
        self._force_qp_fail_log_t = now
        print(f"[3FForceClosure] {msg} (fails={self._force_qp_fail_count})")

    def _publish_force_contact_arrows(self) -> None:
        pts = np.asarray(self._contact_points_world_vis, dtype=np.float64).reshape(
            -1, 3
        )
        qp_pts = np.asarray(
            self._rt_qp_contact_pts_world_vis, dtype=np.float64
        ).reshape(-1, 3)
        frc = np.asarray(self._f_world_vis, dtype=np.float64).reshape(-1, 3)
        if qp_pts.shape[0] == frc.shape[0] and frc.shape[0] > 0:
            pts = qp_pts
        n = min(pts.shape[0], frc.shape[0])
        pts, frc = pts[:n], frc[:n]
        nrms = np.asarray(self._rt_contact_normals_world_vis, dtype=np.float64).reshape(
            -1, 3
        )
        n_n = min(nrms.shape[0], n)
        self._publish_contact_arrows(
            pts,
            nrms[:n_n] if n_n > 0 else np.zeros((0, 3)),
            frc if n > 0 else None,
            clear_forces=(n == 0),
        )

    # ------------------------------------------------------------------
    # RT force stack (adapted from MjEvalGraspControl, 27-DoF FK → 20-DoF τ)
    # ------------------------------------------------------------------

    def _compute_rt_force_torque(self) -> np.ndarray:
        self._compute_force_fk()
        (
            all_link_indices,
            all_points_cad,
            all_normals_cad,
            all_points_link,
            contact_mask,
            n_contacts,
            tau_attract,
        ) = self._rt_detect_contacts()

        enable_approach = self._gen_config.get("enable_fingertip_approach", True)
        tau_approach = (
            self._compute_fingertip_approach_torque(all_link_indices)
            if enable_approach
            else np.zeros(N_HAND_JOINTS)
        )

        enable_force_closure = self._gen_config.get("enable_force_closure", True)
        if n_contacts == 0 or not enable_force_closure:
            self._contact_points_world_vis = np.zeros((0, 3))
            self._f_world_vis = np.zeros((0, 3))
            self._prev_tau_qp *= 0.9  # decay stale hold
            # Pre-contact: approach + light PD close (no force-closure yet).
            return self._clip_hand_tau(
                tau_attract + tau_approach + self._closing_pd_torque()
            )

        tau_qp, solved = self._compute_qp_force_closure(
            all_link_indices[contact_mask],
            all_points_cad[contact_mask],
            all_normals_cad[contact_mask],
            all_points_link[contact_mask],
        )
        if not solved:
            # Do not keep blasting a stale τ_QP into a new contact set.
            self._prev_tau_qp *= 0.5
            tau_qp = self._prev_tau_qp
            # Soft PD keeps a stable squeeze while QP recovers.
            tau_qp = tau_qp + self._closing_pd_torque()
        return self._clip_hand_tau(tau_qp + tau_attract + tau_approach)

    def _compute_force_fk(self) -> None:
        poses = get_fk_link_poses(
            self._q, self._pin_model, self._pin_data_force, self._hand_link_names
        )
        jacs = get_link_Jacobians(
            self._q, self._pin_model, self._pin_data_force, self._hand_link_names
        )
        self._force_poses = {k: v for k, v in zip(self._hand_link_names, poses)}
        # Store hand columns only (20 DoF) for torque mapping / QP limits.
        self._force_Jacobians = {
            k: J[:, self._num_arm :] for k, J in zip(self._hand_link_names, jacs)
        }
        self._cad_in_world_frame = self._T_obj.copy()
        self._world2cad = np.linalg.inv(self._cad_in_world_frame)

    def _hand_J(self, link_name: str) -> np.ndarray:
        return self._force_Jacobians[link_name]

    def _hand_T(self, link_name: str) -> np.ndarray:
        return self._force_poses[link_name]

    def _tf_A2B(self, p_in_A, A2B):
        return (A2B[:3, :3] @ p_in_A.T).T + A2B[:3, 3]

    def _rot_A2B(self, n_in_A, A2B):
        return (A2B[:3, :3] @ n_in_A.T).T

    @staticmethod
    def _compute_tau_from_J_and_f(J, f):
        return (np.transpose(J, (0, 2, 1)) @ f[:, :, np.newaxis])[:, :, 0].sum(axis=0)

    def _rt_detect_contacts(self):
        tf_link_to_obj = [
            self._world2cad @ self._force_poses[name] for name in self._hand_link_names
        ]
        contact_dist_thr = self._gen_config.get("contact_dist_thr", 0.005)
        attract_dist_thr = self._gen_config.get("attract_dist_thr", 0.02)
        max_contacts_per_link = self._gen_config.get("max_contacts_per_link", 3)
        contact_source = self._gen_config.get("contact_source", "object")
        if contact_source not in ("hand", "object"):
            raise ValueError(
                f"contact_source must be 'hand' or 'object', got {contact_source!r}"
            )

        if contact_source == "hand":
            (
                all_link_indices,
                all_points_cad,
                udf_normals_cad,
                all_points_link,
                all_dists,
                hand_normals_cad,
            ) = detect_contacts(
                self._udf_grid,
                self._hand_surface_samples_per_link,
                tf_link_to_obj,
                attract_dist_thr,
                max_contacts_per_link=max_contacts_per_link,
                hand_normals_per_link=self._hand_surface_normals_per_link,
            )
            all_normals_cad = hand_normals_cad
            attract_normals_cad = udf_normals_cad
        else:
            flip_obj_by_hand = self._gen_config.get("flip_obj_normal_by_hand", False)
            flip_obj_by_sign = self._gen_config.get("flip_obj_normal_by_sign", False)
            if flip_obj_by_hand and flip_obj_by_sign:
                raise ValueError(
                    "flip_obj_normal_by_hand and flip_obj_normal_by_sign "
                    "are mutually exclusive."
                )
            hand_normals_cad = None
            contact_signs = None
            if flip_obj_by_hand:
                (
                    all_link_indices,
                    all_points_cad,
                    all_normals_cad,
                    all_points_link,
                    all_dists,
                    hand_normals_cad,
                ) = detect_contacts(
                    self._udf_grid,
                    self._hand_surface_samples_per_link,
                    tf_link_to_obj,
                    attract_dist_thr,
                    max_contacts_per_link=max_contacts_per_link,
                    hand_normals_per_link=self._hand_surface_normals_per_link,
                )
            elif flip_obj_by_sign:
                (
                    all_link_indices,
                    all_points_cad,
                    all_normals_cad,
                    all_points_link,
                    all_dists,
                    contact_signs,
                ) = detect_contacts(
                    self._udf_grid,
                    self._hand_surface_samples_per_link,
                    tf_link_to_obj,
                    attract_dist_thr,
                    max_contacts_per_link=max_contacts_per_link,
                    return_signs=True,
                )
            else:
                (
                    all_link_indices,
                    all_points_cad,
                    all_normals_cad,
                    all_points_link,
                    all_dists,
                ) = detect_contacts(
                    self._udf_grid,
                    self._hand_surface_samples_per_link,
                    tf_link_to_obj,
                    attract_dist_thr,
                    max_contacts_per_link=max_contacts_per_link,
                )
            attract_normals_cad = all_normals_cad
            if len(all_dists) > 0:
                udf_outward = -all_normals_cad
                all_points_cad = all_points_cad - all_dists[:, None] * udf_outward
            if flip_obj_by_hand and len(all_dists) > 0:
                dots = (all_normals_cad * hand_normals_cad).sum(axis=-1)
                flip = dots < 0
                if flip.any():
                    all_normals_cad = all_normals_cad.copy()
                    all_normals_cad[flip] *= -1
            elif flip_obj_by_sign and len(all_dists) > 0:
                flip = contact_signs < 0
                if flip.any():
                    all_normals_cad = all_normals_cad.copy()
                    all_normals_cad[flip] *= -1

        if len(all_dists) > 0:
            contact_mask = all_dists < contact_dist_thr
            attract_mask = ~contact_mask
        else:
            contact_mask = np.array([], dtype=bool)
            attract_mask = np.array([], dtype=bool)

        if len(all_link_indices) > 0 and contact_mask.any():
            qp_pts_world = self._tf_A2B(
                all_points_cad[contact_mask], self._cad_in_world_frame
            )
            self._rt_qp_contact_pts_world_vis = qp_pts_world
            self._rt_contact_normals_world_vis = self._rot_A2B(
                all_normals_cad[contact_mask], self._cad_in_world_frame
            )
        else:
            self._rt_qp_contact_pts_world_vis = np.zeros((0, 3))
            self._rt_contact_normals_world_vis = np.zeros((0, 3))

        attract_stiffness = self._gen_config.get("attract_stiffness", 0.0)
        tau_attract = self._compute_attraction_torque(
            (
                all_link_indices[attract_mask]
                if attract_mask.any()
                else np.array([], dtype=int)
            ),
            all_points_link[attract_mask] if attract_mask.any() else np.zeros((0, 3)),
            (
                attract_normals_cad[attract_mask]
                if attract_mask.any()
                else np.zeros((0, 3))
            ),
            all_dists[attract_mask] if attract_mask.any() else np.array([]),
            attract_stiffness,
        )
        n_contacts = int(contact_mask.sum()) if len(contact_mask) > 0 else 0
        return (
            all_link_indices,
            all_points_cad,
            all_normals_cad,
            all_points_link,
            contact_mask,
            n_contacts,
            tau_attract,
        )

    def _compute_attraction_torque(
        self, link_indices, points_link, normals_cad, dists, stiffness
    ):
        tau = np.zeros(N_HAND_JOINTS)
        if len(dists) == 0 or stiffness == 0.0:
            return tau
        normals_world = self._rot_A2B(normals_cad, self._cad_in_world_frame)
        forces_world = stiffness * dists[:, None] * normals_world
        for i in range(len(link_indices)):
            link_name = self._hand_link_names[link_indices[i]]
            T = self._hand_T(link_name)
            J_link = self._hand_J(link_name)
            J_pt = get_point_Jacobian(
                T[:3, :3], J_link[3:, :], J_link[:3, :], points_link[i]
            )
            tau += J_pt.T @ forces_world[i]
        return tau

    def _compute_fingertip_approach_torque(self, detected_link_indices):
        N_FINGERS = 5
        JOINTS_PER_FINGER = 4
        APPROACH_SPEED = 0.1
        APPROACH_EPS = 0.03
        APPROACH_KD = 0.1
        IKQP_REG = 1e-3
        QDOT_DES_ALPHA = 0.1

        if not hasattr(self, "_finger_link_groups"):
            groups = [[] for _ in range(N_FINGERS)]
            for idx, name in enumerate(self._hand_link_names):
                if not name.startswith("finger_r_link_"):
                    continue
                f_num = int(name.split("_")[3]) - 1
                groups[f_num].append(idx)
            tips = [grp[-1] if grp else None for grp in groups]
            self._finger_link_groups = groups
            self._fingertip_link_idx_per_finger = tips
            self._link_idx_to_finger_id = {
                idx: f for f, idxs in enumerate(groups) for idx in idxs
            }

        contacted_fingers = set()
        for idx in detected_link_indices:
            f = self._link_idx_to_finger_id.get(int(idx))
            if f is not None:
                contacted_fingers.add(f)

        non_contacting = [
            f
            for f in range(N_FINGERS)
            if f not in contacted_fingers
            and self._fingertip_link_idx_per_finger[f] is not None
        ]
        if not non_contacting:
            self._approach_qdot_des_avg = np.zeros(N_HAND_JOINTS)
            return np.zeros(N_HAND_JOINTS)

        task_space_plans = []
        for finger_id in non_contacting:
            tip_idx = self._fingertip_link_idx_per_finger[finger_id]
            tip_name = self._hand_link_names[tip_idx]
            T_tip = self._hand_T(tip_name)
            J_tip_lin = self._hand_J(tip_name)[3:, :]
            tip_pos_cad = (
                self._world2cad[:3, :3] @ T_tip[:3, 3] + self._world2cad[:3, 3]
            )
            dist, normal = self._udf_grid.query(tip_pos_cad.reshape(1, 3))
            surface_pt_cad = tip_pos_cad - dist[0] * normal[0]
            surface_pt_world = (
                self._cad_in_world_frame[:3, :3] @ surface_pt_cad
                + self._cad_in_world_frame[:3, 3]
            )
            xdot_des = goto_x_des_const_then_linear_clamped(
                T_tip[:3, 3],
                surface_pt_world,
                APPROACH_SPEED,
                APPROACH_EPS,
                1.0 / self._ctrl_freq,
            )
            task_space_plans.append((1.0, J_tip_lin, xdot_des))

        P = IKQP_REG * np.eye(N_HAND_JOINTS)
        g = np.zeros(N_HAND_JOINTS)
        for w, J, xd in task_space_plans:
            P += w * (J.T @ J)
            g += -w * (xd @ J)
        qdot_des_raw = np.linalg.solve(P, -g)
        qdot_des = (
            self._approach_qdot_des_avg * (1 - QDOT_DES_ALPHA)
            + qdot_des_raw * QDOT_DES_ALPHA
        )
        self._approach_qdot_des_avg = qdot_des

        qdot = self._qd[self._num_arm :]
        mask = np.zeros(N_HAND_JOINTS)
        for finger_id in non_contacting:
            s = finger_id * JOINTS_PER_FINGER
            mask[s : s + JOINTS_PER_FINGER] = 1.0
        return APPROACH_KD * mask * (qdot_des - qdot)

    def _solve_contact_force_qp(
        self,
        c_link_indices,
        c_points_cad,
        c_normals_cad,
        c_points_link,
        *,
        enable_achievable_force=None,
        min_fn=None,
        quiet=False,
    ):
        n_contacts = len(c_link_indices)
        J_contact = np.zeros((n_contacts, 3, N_HAND_JOINTS))
        for i in range(n_contacts):
            link_name = self._hand_link_names[c_link_indices[i]]
            T = self._hand_T(link_name)
            J_link = self._hand_J(link_name)
            J_contact[i] = get_point_Jacobian(
                T[:3, :3], J_link[3:, :], J_link[:3, :], c_points_link[i]
            )

        n = c_normals_cad[np.newaxis, :, :]
        c = c_points_cad[np.newaxis, :, :]
        Jr = J_contact[np.newaxis, :, :, :]
        Jo, Jr_contact, n, t1, t2 = compute_contact_frame(n, c, Jr)
        Jo, Jr_contact = Jo[0], Jr_contact[0]
        n, t1, t2 = n[0], t1[0], t2[0]

        torque_limit = self._hand_torque_limit()
        if min_fn is None:
            min_fn = self._gen_config.get("min_fn", 5.0)
        mu = self._gen_config.get("mu", 0.3)
        null_space_margin = self._gen_config.get("null_space_margin", 0.01)
        enable_torque_limit = self._gen_config.get("enable_torque_limit", True)
        if enable_achievable_force is None:
            enable_achievable_force = self._gen_config.get(
                "enable_achievable_force", False
            )
        actuated_mask = c_link_indices != self._palm_link_idx

        sigma_p = float(self._gen_config.get("eps_translation", 0.0))
        sigma_th = float(self._gen_config.get("eps_rotation", 0.0))
        if self._gen_config.get("eps_lever_only", False):
            sigma_p = 0.0
        if sigma_p > 0.0 or sigma_th > 0.0:
            cross = np.cross(c_points_cad - self._obj_center_in_cad_frame, n)
            contact_epsilon = np.sqrt(
                sigma_p**2 + sigma_th**2 * (cross**2).sum(axis=-1)
            )
            eps_min = float(self._gen_config.get("eps_min", 1e-3))
            if eps_min > 0.0:
                contact_epsilon = np.maximum(contact_epsilon, eps_min)
        else:
            contact_epsilon = None
        eps_weight = float(self._gen_config.get("eps_weight", 1.0))

        f, solved = generate_contact_forces(
            Jo,
            Jr_contact,
            torque_limit,
            min_fn,
            mu,
            null_space_margin,
            enable_torque_limit=enable_torque_limit,
            enable_achievable_force=enable_achievable_force,
            actuated_mask=actuated_mask,
            contact_epsilon=contact_epsilon,
            eps_weight=eps_weight,
            verbose=False,
        )
        if solved:
            f_cad = f[:, 0:1] * n + f[:, 1:2] * t1 + f[:, 2:3] * t2
            f_world = self._rot_A2B(f_cad, self._cad_in_world_frame)
        else:
            if not quiet:
                self._log_force_qp_fail(
                    f"force-closure OSQP infeasible (P={n_contacts}, "
                    f"achievable={enable_achievable_force}, min_fn={min_fn})"
                )
            f_cad = np.zeros((n_contacts, 3))
            f_world = np.zeros((n_contacts, 3))
        return J_contact, f_cad, f_world, solved

    def _compute_hand_pts_world(self, c_link_indices, c_points_link):
        n = len(c_link_indices)
        pts = np.zeros((n, 3))
        for i in range(n):
            T = self._hand_T(self._hand_link_names[c_link_indices[i]])
            pts[i] = T[:3, :3] @ c_points_link[i] + T[:3, 3]
        return pts

    def _compute_qp_force_closure(
        self, c_link_indices, c_points_cad, c_normals_cad, c_points_link
    ):
        """Solve force-closure; soften constraints once if the first QP fails."""
        J_contact, f_cad, f_world, solved = self._solve_contact_force_qp(
            c_link_indices, c_points_cad, c_normals_cad, c_points_link
        )
        if not solved:
            # Retry: drop achievable-force nullspace (often infeasible with RT
            # contacts) and lower min_fn.
            soft_min = 0.5 * float(self._gen_config.get("min_fn", 5.0))
            J_contact, f_cad, f_world, solved = self._solve_contact_force_qp(
                c_link_indices,
                c_points_cad,
                c_normals_cad,
                c_points_link,
                enable_achievable_force=False,
                min_fn=max(soft_min, 1.0),
                quiet=True,
            )
            if solved:
                now = time.time()
                if now - self._force_qp_fail_log_t >= 1.0:
                    self._force_qp_fail_log_t = now
                    print(
                        "[3FForceClosure] force-closure recovered "
                        "with softened constraints"
                    )
                self._force_qp_fail_count = 0

        if solved:
            self._contact_points_world_vis = self._compute_hand_pts_world(
                c_link_indices, c_points_link
            )
            self._f_world_vis = f_world
            tau_qp = self._clip_hand_tau(
                self._compute_tau_from_J_and_f(J_contact, f_world)
            )
            self._prev_tau_qp = tau_qp
            self._force_qp_fail_count = 0
        else:
            tau_qp = np.zeros(N_HAND_JOINTS, dtype=np.float64)
        return tau_qp, solved
