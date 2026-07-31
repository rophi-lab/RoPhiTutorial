"""Hand-only reactive grasp controller for MuJoCo eval (RoPhiTutorial).

Ports the *exact* control algorithm from ReactiveGrasp
``scripts/planning/offline/mj_eval/MjMeshesGraspEvalEnv.py`` into a
``BaseController`` that subscribes to hand joint meas + object bb2world pose
and publishes ``JointCtrl`` tau_ff plus contact-force NamedVecLists.

Contact detection defaults to Warp ``UDFGrid`` on GPU (``cuda:0``), with CPU
``MeshDistanceField`` as fallback if Warp/CUDA is unavailable.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pinocchio as pin
import trimesh
from omegaconf import DictConfig

from controller.BaseController import BaseController
from controller.grasping.utils.Robotis5FGraspData import Robotis5FGraspData
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.NamedVecListData import NamedVecListData
from utils.grasping.distance_field import build_distance_field, detect_contacts
from utils.grasping.force_gen import compute_contact_frame, generate_contact_forces
from utils.grasping.GraspSqueezeParamsGenerator import GraspSqueezeParamsGenerator
from utils.grasping.link_surface_sampler import load_link_surface_samples
from utils.lie.kinematics import get_point_Jacobian
from utils.pinocchio.getter import get_fk_link_poses, get_link_Jacobians
from utils.velocity_fields.grasping import goto_x_des_const_then_linear_clamped

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

HAND_LINK_NAMES = [
    "finger_r_link_1_thumb1",
    "finger_r_link_1_thumb2",
    "finger_r_link_1_thumb3",
    "finger_r_link_1_thumb4",
    "finger_r_link_2_index1",
    "finger_r_link_2_index2",
    "finger_r_link_2_index3",
    "finger_r_link_2_index4",
    "finger_r_link_3_middle1",
    "finger_r_link_3_middle2",
    "finger_r_link_3_middle3",
    "finger_r_link_3_middle4",
    "finger_r_link_4_ring1",
    "finger_r_link_4_ring2",
    "finger_r_link_4_ring3",
    "finger_r_link_4_ring4",
    "finger_r_link_5_little1",
    "finger_r_link_5_little2",
    "finger_r_link_5_little3",
    "finger_r_link_5_little4",
    "palm",
]

N_HAND_JOINTS = 20


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_PROJECT_ROOT, p))


class MjEvalGraspControl(BaseController):
    """Reactive grasp control ported from MjMeshesGraspEvalEnv."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._config = config
        self._control_type = config.get(
            "control_type", "real_time_feedforward_force_control"
        )

        self._joint_meas_channel = config["sub_manager"].get(
            "joint_meas_channel", "sim_hand_joint_meas"
        )
        self._obj_pose_channel = config["sub_manager"].get(
            "named_vec_list_channel", "sim_obj_pose_bb2world"
        )
        self._contact_forces_channel = config["pub_manager"].get(
            "named_vec_list_channel_1", "sim_contact_forces_in_world"
        )
        self._net_force_channel = config["pub_manager"].get(
            "named_vec_list_channel_2", "sim_net_force_arrow"
        )

        self._q = np.zeros(N_HAND_JOINTS)
        self._qd = np.zeros(N_HAND_JOINTS)
        self._obj_pos = np.zeros(3)
        self._obj_rot = np.eye(3)
        self._data_ready = False
        self._state = "idle"  # "idle" or "control"
        self._grav_comp_enabled = False
        self._physics_params_received = False
        self._grasp_idx = int(config.get("grasp_idx", 0))

        self._object_mass = 0.0
        self._com_cad = np.zeros(3)
        self._gravity_force_world = None

        self._prev_tau_qp = np.zeros(N_HAND_JOINTS)
        self._approach_qdot_des_avg = np.zeros(N_HAND_JOINTS)
        self._dict_poses = {}
        self._dict_Jacobians = {}
        self._contact_points_world_vis = np.zeros((0, 3))
        self._f_world_vis = np.zeros((0, 3))
        self._ctrl_loop_dts = []

        # Cached offline-control frames (invalidated on grasp change).
        self._ff_grasp_idx = -1
        self._tsi_grasp_idx = -1

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def initialize(self):
        cfg = self._config
        gen_cfg_yaml = dict(cfg.get("grasp_squeeze_params_generator_config", {}) or {})
        self._gen_config = dict(gen_cfg_yaml)

        urdf_path = _resolve_path(
            gen_cfg_yaml.get(
                "urdf_path",
                cfg.get("urdf_path", "assets/scene/robotis_5f/robotis_5f.urdf"),
            )
        )
        mesh_path_cfg = gen_cfg_yaml.get(
            "mesh_path",
            cfg.get("mesh_path", "assets/scene/robotis_5f"),
        )
        if isinstance(mesh_path_cfg, dict):
            mesh_paths = [_resolve_path(p) for p in mesh_path_cfg.values()]
        else:
            mesh_paths = [_resolve_path(str(mesh_path_cfg))]

        self._pin_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_paths)
        self._pin_model = self._pin_robot.model
        self._pin_data = self._pin_robot.data

        self._gen_config["_urdf_path"] = urdf_path
        hand_mesh_dir = _resolve_path(mesh_paths[0])
        self._gen_config["_mesh_dir"] = hand_mesh_dir

        obj_mesh_dir = _resolve_path(
            cfg.get("obj_mesh_dir", gen_cfg_yaml.get("mesh_path_dir", ""))
        )
        mesh_obj_path = cfg.get("obj_mesh_path", None)
        if mesh_obj_path is not None:
            mesh_obj_path = _resolve_path(mesh_obj_path)
        else:
            mesh_obj_path = os.path.join(obj_mesh_dir, "textured_mesh.obj")
        if not os.path.isfile(mesh_obj_path):
            raise FileNotFoundError(f"Object mesh not found: {mesh_obj_path}")

        mesh = trimesh.load(mesh_obj_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        self._mesh = mesh
        self._obj_center_in_cad_frame = np.asarray(self._mesh.centroid)

        com_cad_cfg = cfg.get("object_com_cad", None)
        if com_cad_cfg is not None:
            self._com_cad = np.asarray(com_cad_cfg, dtype=float)
        else:
            self._com_cad = np.asarray(
                mesh.center_mass if hasattr(mesh, "center_mass") else mesh.centroid
            )

        # Env publishes bb2world; convert to cad2world via oriented_bounds.
        self._cad_to_bb, _ = trimesh.bounds.oriented_bounds(mesh)
        self._bb_to_cad = np.linalg.inv(self._cad_to_bb)

        grasp_predictions_path = cfg.get("grasp_predictions_path", None)
        if grasp_predictions_path is None:
            grasp_predictions_path = os.path.join(obj_mesh_dir, "grasp_predictions")
        else:
            grasp_predictions_path = _resolve_path(grasp_predictions_path)
        self._grasp_data = Robotis5FGraspData(grasp_predictions_path)
        self._n_grasps = len(self._grasp_data.get_grasp_poses_in_cad_frame())
        if self._grasp_idx < 0 or self._grasp_idx >= self._n_grasps:
            raise ValueError(
                f"grasp_idx={self._grasp_idx} out of range [0, {self._n_grasps})"
            )

        # Offline squeeze forces (needed for feedforward / impedance modes).
        needs_offline = self._control_type in (
            "feedforward_force_control",
            "task_space_impedance_control",
        ) or bool(cfg.get("generate_offline_forces", False))
        self._grasp_squeeze_params_generator = GraspSqueezeParamsGenerator(
            pin_robot=self._pin_robot,
            mesh=self._mesh,
            grasp_data=self._grasp_data,
            config=self._gen_config,
        )
        if needs_offline:
            self._f_in_cad_frame = (
                self._grasp_squeeze_params_generator.generate_contact_forces()
            )
            self._contact_points_in_cad_frame = (
                self._grasp_squeeze_params_generator.get_contact_points_in_cad_frame()
            )
        else:
            self._f_in_cad_frame = None
            self._contact_points_in_cad_frame = (
                self._grasp_data.get_contact_points_in_cad_frame()
            )

        self._k_for_contact_force_control = float(
            self._gen_config.get("k_for_contact_force_control", 5.0)
        )
        if self._f_in_cad_frame is not None:
            self._target_contact_points_in_cad_frame = (
                self._contact_points_in_cad_frame
                + (1.0 / self._k_for_contact_force_control) * self._f_in_cad_frame
            )
        else:
            self._target_contact_points_in_cad_frame = None

        # Distance field: GPU UDF by default (Warp), CPU mesh fallback.
        udf_res = int(self._gen_config.get("udf_resolution", 128))
        udf_pad = float(self._gen_config.get("udf_padding", 0.02))
        udf_device = str(
            self._gen_config.get(
                "udf_device", cfg.get("udf_device", "cuda:0")
            )
        )
        udf_backend = str(
            self._gen_config.get(
                "distance_field_backend",
                cfg.get("distance_field_backend", "auto"),
            )
        )
        self._udf_grid, self._distance_field_backend = build_distance_field(
            self._mesh,
            resolution=udf_res,
            padding=udf_pad,
            device=udf_device,
            backend=udf_backend,
        )

        # Hand surface samples (density + normals, same as env.initialize).
        density = self._gen_config.get("surface_density_per_cm2", 1.0)
        finger_links = [l for l in HAND_LINK_NAMES if l != "palm"]
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

        gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        object_mass = cfg.get("object_mass", None)
        self._object_mass = float(object_mass) if object_mass is not None else 0.0
        if self._object_mass > 0:
            self._gravity_force_world = self._object_mass * gravity

        print(
            f"[MjEvalGraspControl] Initialized "
            f"(control_type={self._control_type}, "
            f"grasp_idx={self._grasp_idx}/{self._n_grasps}, "
            f"offline_forces={needs_offline}, "
            f"distance_field={self._distance_field_backend})"
        )

    # ------------------------------------------------------------------
    # Keys / BaseController I/O
    # ------------------------------------------------------------------

    def _handle_key(self, key):
        if key == "g":
            self._grav_comp_enabled = not self._grav_comp_enabled
            state = "ON" if self._grav_comp_enabled else "OFF"
            print(f"[MjEvalGraspControl] Hand gravity compensation: {state}")
        elif key == "v":
            if self._state == "control":
                self._state = "idle"
                print("[MjEvalGraspControl] Reactive grasp control: OFF")
            else:
                self._state = "control"
                print("[MjEvalGraspControl] Reactive grasp control: ON")
        else:
            super()._handle_key(key)

    def _check_and_get_data_from_que(self):
        if self._joint_meas_channel in self.intr_sub_que_dict:
            que = self.intr_sub_que_dict[self._joint_meas_channel]
            while not que.empty():
                data = que.get_nowait()
                self._q[:] = data.get_q()[:N_HAND_JOINTS]
                self._qd[:] = data.get_qd()[:N_HAND_JOINTS]
                self._data_ready = True

        if self._obj_pose_channel in self.extr_sub_que_dict:
            que = self.extr_sub_que_dict[self._obj_pose_channel]
            while not que.empty():
                data = que.get_nowait()
                _, name_list, vec_list = data.get_data()
                if len(vec_list) > 0:
                    pose_12d = vec_list[0]
                    self._obj_pos[:] = pose_12d[:3]
                    self._obj_rot = pose_12d[3:12].reshape(3, 3).copy()

                if name_list is not None and "physics_params" in name_list:
                    idx = name_list.index("physics_params")
                    p = vec_list[idx]
                    mass = float(p[0])
                    com_cad = p[1:4].copy()
                    gravity = p[4:7].copy()
                    if not self._physics_params_received:
                        self._physics_params_received = True
                        self._object_mass = mass
                        self._com_cad = com_cad
                        if mass > 0:
                            self._gravity_force_world = mass * gravity
                        else:
                            self._gravity_force_world = None
                        print(
                            f"[MjEvalGraspControl] MuJoCo physics: "
                            f"mass={mass:.4f}kg, com_cad={com_cad}, "
                            f"gravity={gravity}"
                        )

    def _update(self):
        if not self._data_ready:
            return

        t0 = time.time()

        tau = np.zeros(N_HAND_JOINTS)
        if self._grav_comp_enabled:
            # Need FK data for pinocchio gravity (q only); no link FK required.
            tau += pin.computeGeneralizedGravity(
                self._pin_model, self._pin_data, self._q
            )

        if self._state == "control":
            ctrl_data = JointCtrlData(num_joints=N_HAND_JOINTS)
            self._set_ctrl_data_overwrite(ctrl_data)
            _, _, _, tau_ff, _, _ = ctrl_data.get_data()
            tau = tau + tau_ff

        out = JointCtrlData(num_joints=N_HAND_JOINTS)
        out.set_data(
            t=self._cur_time,
            q_des=np.zeros(N_HAND_JOINTS),
            qd_des=np.zeros(N_HAND_JOINTS),
            tau_ff=tau,
            kp=np.zeros(N_HAND_JOINTS),
            kd=np.zeros(N_HAND_JOINTS),
        )
        if self.ctrl_pub_que is not None:
            self.ctrl_pub_que.put(out)

        self._publish_contact_forces()

        dt = time.time() - t0
        self._ctrl_loop_dts.append(dt)
        if len(self._ctrl_loop_dts) % 100 == 0:
            mean_dt = float(np.mean(self._ctrl_loop_dts[-100:]))
            if mean_dt > 0:
                print(
                    f"[MjEvalGraspControl] ctrl dt: {mean_dt * 1000:.1f}ms "
                    f"({1.0 / mean_dt:.0f} Hz)"
                )

    def _publish_contact_forces(self):
        """Publish QP contact points + forces (+ normals) every control tick.

        Channel rows (vec_dim=6):
          - ``"0"``, ``"1"``, … : contact point + force  [p | f] in world
          - ``"n_0"``, …         : contact point + normal [p | n] in world
          - ``"empty"``          : clear signal when there are no contacts
        """
        if self.pub_que_dict is None:
            return

        pts = np.asarray(self._contact_points_world_vis, dtype=np.float64).reshape(
            -1, 3
        )
        # Prefer QP optimization contact points when count matches force rows.
        qp_pts = np.asarray(
            getattr(self, "_rt_qp_contact_pts_world_vis", np.zeros((0, 3))),
            dtype=np.float64,
        ).reshape(-1, 3)
        frc = np.asarray(self._f_world_vis, dtype=np.float64).reshape(-1, 3)
        if qp_pts.shape[0] == frc.shape[0] and frc.shape[0] > 0:
            pts = qp_pts
        if pts.shape[0] != frc.shape[0]:
            n = min(pts.shape[0], frc.shape[0])
            pts = pts[:n]
            frc = frc[:n]

        nrm = np.asarray(
            getattr(self, "_rt_contact_normals_world_vis", np.zeros((0, 3))),
            dtype=np.float64,
        ).reshape(-1, 3)

        if self._contact_forces_channel in self.pub_que_dict:
            if pts.shape[0] == 0:
                name_list = ["empty"]
                vecs = np.zeros((1, 6))
            else:
                name_list = [str(i) for i in range(pts.shape[0])]
                vecs = np.hstack([pts, frc])
                # Append normals (same contact points) when available.
                n_n = min(nrm.shape[0], pts.shape[0])
                if n_n > 0:
                    name_list = name_list + [f"n_{i}" for i in range(n_n)]
                    vecs = np.vstack(
                        [vecs, np.hstack([pts[:n_n], nrm[:n_n]])]
                    )
            cf_data = NamedVecListData(
                name=self._contact_forces_channel,
                num_vecs=len(name_list),
                vec_dim=6,
            )
            cf_data.set_data(t=self._cur_time, name_list=name_list, vec_list=vecs)
            self.pub_que_dict[self._contact_forces_channel].put(cf_data)

        if self._net_force_channel in self.pub_que_dict:
            if pts.shape[0] == 0 or not hasattr(self, "_cad_in_world_frame"):
                name_list = ["empty"]
                vecs = np.zeros((1, 6))
            else:
                cad_origin_world = self._cad_in_world_frame[:3, 3]
                net_f = frc.sum(axis=0)
                name_list = ["net"]
                vecs = np.concatenate([cad_origin_world, net_f]).reshape(1, 6)
            nf_data = NamedVecListData(
                name=self._net_force_channel,
                num_vecs=len(name_list),
                vec_dim=6,
            )
            nf_data.set_data(t=self._cur_time, name_list=name_list, vec_list=vecs)
            self.pub_que_dict[self._net_force_channel].put(nf_data)

    # ------------------------------------------------------------------
    # FK / frames
    # ------------------------------------------------------------------

    def _compute_variables(self):
        list_link_names = list(HAND_LINK_NAMES)
        link_poses = get_fk_link_poses(
            self._q, self._pin_model, self._pin_data, list_link_names
        )
        list_Jacobians = get_link_Jacobians(
            self._q, self._pin_model, self._pin_data, list_link_names
        )
        self._dict_poses = {k: v for k, v in zip(list_link_names, link_poses)}
        self._dict_Jacobians = {k: v for k, v in zip(list_link_names, list_Jacobians)}

    def _get_cad_in_world_frame(self):
        """bb2world from subscriber → cad2world via oriented_bounds."""
        T_bb2world = np.eye(4)
        T_bb2world[:3, :3] = self._obj_rot
        T_bb2world[:3, 3] = self._obj_pos
        # env: T_bb2world = T_cad2world @ _bb_to_cad
        # ⇒ T_cad2world = T_bb2world @ _cad_to_bb
        return T_bb2world @ self._cad_to_bb

    def _compute_contact_link_poses_and_Jacobians(
        self,
        contact_points_in_link_frame,  # (P, 3)
        link_idxs_of_contact_points,  # (P,)
    ):
        P = len(link_idxs_of_contact_points)
        J_contact_points_in_world_frame = np.zeros((P, 3, N_HAND_JOINTS))
        tf_link2world = np.zeros((P, 4, 4))
        for contact_idx, link_idx in enumerate(link_idxs_of_contact_points):
            link_name = self._grasp_data.get_mapper_idx_to_link_name()[link_idx]
            link_pose = self._dict_poses[link_name]
            link_Jacobian = self._dict_Jacobians[link_name]
            J_contact_points_in_world_frame[contact_idx, :, :] = get_point_Jacobian(
                link_pose[0:3, 0:3],
                link_Jacobian[3:, :],
                link_Jacobian[:3, :],
                contact_points_in_link_frame[contact_idx],
            )
            tf_link2world[contact_idx, :, :] = link_pose
        return tf_link2world, J_contact_points_in_world_frame

    def _convert_f_in_cad_to_world_frame(self, f_in_cad_frame, tf_cad2world):
        return (tf_cad2world[:3, :3] @ f_in_cad_frame.T).T

    def _get_target_cp_in_world_frame(self, target_cp_in_cad, tf_cad2world):
        return (tf_cad2world[:3, :3] @ target_cp_in_cad.T).T + tf_cad2world[:3, 3]

    def _get_current_cp_in_world_frame(
        self, contact_points_in_link_frame, tf_link2world
    ):
        return (
            tf_link2world[:, :3, :3] @ contact_points_in_link_frame.reshape(-1, 3, 1)
        ).reshape(-1, 3) + tf_link2world[:, :3, 3]

    def _tf_A2B(self, p_in_A, A2B):
        return (A2B[:3, :3] @ p_in_A.T).T + A2B[:3, 3]

    def _rot_A2B(self, n_in_A, A2B):
        return (A2B[:3, :3] @ n_in_A.T).T

    @staticmethod
    def _compute_tau_from_J_and_f(J, f):
        return (
            np.transpose(J, (0, 2, 1)) @ f[:, :, np.newaxis]
        )[:, :, 0].sum(axis=0)

    def _load_grasp_contact_data(self):
        if self._f_in_cad_frame is None:
            raise RuntimeError(
                "Offline contact forces not generated. Set control_type to an "
                "offline mode or generate_offline_forces=true."
            )
        P_i = int(self._grasp_data.get_num_contacts_per_grasp()[self._grasp_idx])
        link_idxs = self._grasp_data.get_link_idxs_of_contact_points()[
            self._grasp_idx, :P_i
        ]
        contact_points_in_link_frame = (
            self._grasp_data.get_contact_points_in_link_frame()[
                self._grasp_idx, :P_i
            ]
        )
        f_in_cad_frame = self._f_in_cad_frame[self._grasp_idx, :P_i]
        return P_i, link_idxs, contact_points_in_link_frame, f_in_cad_frame

    def _set_torque_only_ctrl(self, ctrl_data, tau_ff):
        ctrl_data.set_data(
            t=self._cur_time,
            q_des=np.zeros(N_HAND_JOINTS),
            qd_des=np.zeros(N_HAND_JOINTS),
            tau_ff=tau_ff,
            kp=np.zeros(N_HAND_JOINTS),
            kd=np.zeros(N_HAND_JOINTS),
        )

    # ------------------------------------------------------------------
    # Control dispatch
    # ------------------------------------------------------------------

    def _set_ctrl_data_overwrite(self, ctrl_data):
        """Dispatch to the active control-type handler (env: _set_ctrl_data_overwite)."""
        self._compute_variables()

        if self._control_type == "feedforward_force_control":
            self._ctrl_feedforward_force(ctrl_data)
        elif self._control_type == "task_space_impedance_control":
            self._ctrl_task_space_impedance(ctrl_data)
        elif self._control_type == "real_time_feedforward_force_control":
            self._ctrl_real_time_feedforward_force(ctrl_data)
        elif self._control_type == "real_time_task_space_impedance":
            self._ctrl_real_time_task_space_impedance(ctrl_data)
        else:
            raise ValueError(f"Unknown control_type: {self._control_type!r}")

    # ------------------------------------------------------------------
    # Feedforward force control
    # ------------------------------------------------------------------

    def _ctrl_feedforward_force(self, ctrl_data):
        P_i, link_idxs, cp_link, f_cad = self._load_grasp_contact_data()
        self._cad_in_world_frame = self._get_cad_in_world_frame()
        self._world2cad = np.linalg.inv(self._cad_in_world_frame)

        tf_link2world, J_cp = self._compute_contact_link_poses_and_Jacobians(
            cp_link, link_idxs
        )

        if (
            not hasattr(self, "_ff_f_in_link_frame")
            or self._ff_grasp_idx != self._grasp_idx
        ):
            R_cad = self._cad_in_world_frame[:3, :3]
            self._ff_f_in_link_frame = np.zeros((P_i, 3))
            for i in range(P_i):
                R_link = tf_link2world[i, :3, :3]
                self._ff_f_in_link_frame[i] = R_link.T @ R_cad @ f_cad[i]
            self._ff_grasp_idx = self._grasp_idx

        f_world = (
            tf_link2world[:P_i, :3, :3] @ self._ff_f_in_link_frame[:, :, np.newaxis]
        )[:, :, 0]

        self._set_torque_only_ctrl(
            ctrl_data, self._compute_tau_from_J_and_f(J_cp, f_world)
        )

        self._f_in_cad_frame_vis = f_cad
        self._contact_points_in_cad_frame_vis = self._contact_points_in_cad_frame[
            self._grasp_idx, :P_i
        ]
        self._contact_points_world_vis = self._get_current_cp_in_world_frame(
            cp_link, tf_link2world
        )
        self._f_world_vis = f_world

    # ------------------------------------------------------------------
    # Task-space impedance control
    # ------------------------------------------------------------------

    def _ctrl_task_space_impedance(self, ctrl_data):
        P_i, link_idxs, cp_link, f_cad = self._load_grasp_contact_data()

        tf_link2world, J_cp = self._compute_contact_link_poses_and_Jacobians(
            cp_link, link_idxs
        )

        if (
            not hasattr(self, "_tsi_target_cp_in_world")
            or self._tsi_grasp_idx != self._grasp_idx
        ):
            self._cad_in_world_frame = self._get_cad_in_world_frame()
            self._world2cad = np.linalg.inv(self._cad_in_world_frame)
            target_cp_in_cad = self._target_contact_points_in_cad_frame[
                self._grasp_idx, :P_i
            ]
            self._tsi_target_cp_in_world = self._get_target_cp_in_world_frame(
                target_cp_in_cad, self._cad_in_world_frame
            )
            self._tsi_f_in_world = self._convert_f_in_cad_to_world_frame(
                f_cad, self._cad_in_world_frame
            )
            self._tsi_grasp_idx = self._grasp_idx

        target_cp_world = self._tsi_target_cp_in_world
        current_cp_world = self._get_current_cp_in_world_frame(cp_link, tf_link2world)
        control_f_world = self._k_for_contact_force_control * (
            target_cp_world - current_cp_world
        )

        self._set_torque_only_ctrl(
            ctrl_data, self._compute_tau_from_J_and_f(J_cp, control_f_world)
        )

        self._f_in_cad_frame_vis = f_cad
        self._contact_points_in_cad_frame_vis = self._contact_points_in_cad_frame[
            self._grasp_idx, :P_i
        ]
        self._target_cp_in_world_frame_vis = target_cp_world
        self._current_cp_in_world_frame_vis = current_cp_world
        self._contact_points_world_vis = current_cp_world
        self._f_world_vis = control_f_world

    # ------------------------------------------------------------------
    # Real-time contact detection (UDF/MeshDistanceField + attraction)
    # ------------------------------------------------------------------

    def _rt_detect_contacts(self):
        """Run contact detection, compute attraction torque, and split masks.

        Ported from MjMeshesGraspEvalEnv._rt_detect_contacts.
        MeshDistanceField.query returns outward normals (like ∇UDF); detect_contacts
        returns inward normals (−outward), matching −∇UDF.
        """
        self._cad_in_world_frame = self._get_cad_in_world_frame()
        self._world2cad = np.linalg.inv(self._cad_in_world_frame)

        tf_link_to_obj = [
            self._world2cad @ self._dict_poses[name] for name in self._hand_link_names
        ]

        contact_dist_thr = self._gen_config.get("contact_dist_thr", 0.005)
        attract_dist_thr = self._gen_config.get("attract_dist_thr", 0.02)
        max_contacts_per_link = self._gen_config.get("max_contacts_per_link", 3)

        contact_source = self._gen_config.get("contact_source", None)
        if contact_source is None:
            contact_source = (
                "hand"
                if self._gen_config.get("use_hand_contacts", False)
                else "object"
            )
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

        self._update_rt_vis_data(
            tf_link_to_obj,
            contact_dist_thr,
            all_link_indices,
            all_points_cad,
            all_normals_cad,
            contact_mask,
        )

        attract_stiffness = self._gen_config.get("attract_stiffness", 5.0)
        tau_attract = self._compute_attraction_torque(
            all_link_indices[attract_mask]
            if attract_mask.any()
            else np.array([], dtype=int),
            all_points_link[attract_mask]
            if attract_mask.any()
            else np.zeros((0, 3)),
            attract_normals_cad[attract_mask]
            if attract_mask.any()
            else np.zeros((0, 3)),
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

    def _update_rt_vis_data(
        self,
        tf_link_to_obj,
        contact_dist_thr,
        all_link_indices,
        all_points_cad,
        all_normals_cad,
        contact_mask,
    ):
        """Cache RT contact points/normals for force publishing (no MuJoCo viewer)."""
        del tf_link_to_obj, contact_dist_thr  # unused without viewer
        if len(all_link_indices) > 0 and contact_mask.any():
            qp_pts_world = self._tf_A2B(
                all_points_cad[contact_mask], self._cad_in_world_frame
            )
            contact_normals_world = self._rot_A2B(
                all_normals_cad[contact_mask], self._cad_in_world_frame
            )
            self._rt_qp_contact_pts_world_vis = qp_pts_world
            self._rt_contact_pts_world_vis = qp_pts_world
            self._rt_contact_normals_world_vis = contact_normals_world
        else:
            self._rt_contact_pts_world_vis = np.zeros((0, 3))
            self._rt_contact_normals_world_vis = np.zeros((0, 3))

    def _compute_fingertip_approach_torque(self, detected_link_indices):
        """TSVF + IKQP approach torque for non-contacting fingertips."""
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

        if not hasattr(self, "_approach_qdot_des_avg"):
            self._approach_qdot_des_avg = np.zeros(N_HAND_JOINTS)

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

        sim_dt = 1.0 / self._ctrl_freq

        task_space_plans = []
        for finger_id in non_contacting:
            tip_idx = self._fingertip_link_idx_per_finger[finger_id]
            tip_name = self._hand_link_names[tip_idx]
            T_tip = self._dict_poses[tip_name]
            J_tip_lin = self._dict_Jacobians[tip_name][3:, :]

            tip_pos_cad = (
                self._world2cad[:3, :3] @ T_tip[:3, 3] + self._world2cad[:3, 3]
            )

            dist, normal = self._udf_grid.query(tip_pos_cad.reshape(1, 3))
            # MeshDistanceField / UDF: outward normal; unsigned dist.
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
                sim_dt,
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

        qdot = self._qd[:N_HAND_JOINTS]
        mask = np.zeros(N_HAND_JOINTS)
        for finger_id in non_contacting:
            s = finger_id * JOINTS_PER_FINGER
            mask[s : s + JOINTS_PER_FINGER] = 1.0

        return APPROACH_KD * mask * (qdot_des - qdot)

    def _ctrl_real_time_feedforward_force(self, ctrl_data):
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
            self._set_torque_only_ctrl(ctrl_data, tau_attract + tau_approach)
        else:
            tau_qp = self._compute_qp_force_closure(
                all_link_indices[contact_mask],
                all_points_cad[contact_mask],
                all_normals_cad[contact_mask],
                all_points_link[contact_mask],
            )
            self._set_torque_only_ctrl(
                ctrl_data, tau_qp + tau_attract + tau_approach
            )

    def _ctrl_real_time_task_space_impedance(self, ctrl_data):
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
            self._set_torque_only_ctrl(ctrl_data, tau_attract + tau_approach)
        else:
            tau_imp = self._compute_qp_impedance_torque(
                all_link_indices[contact_mask],
                all_points_cad[contact_mask],
                all_normals_cad[contact_mask],
                all_points_link[contact_mask],
            )
            self._set_torque_only_ctrl(
                ctrl_data, tau_imp + tau_attract + tau_approach
            )

    def _compute_attraction_torque(
        self, link_indices, points_link, normals_cad, dists, stiffness
    ):
        tau = np.zeros(N_HAND_JOINTS)
        if len(dists) == 0:
            return tau

        normals_world = self._rot_A2B(normals_cad, self._cad_in_world_frame)
        forces_world = stiffness * dists[:, None] * normals_world

        for i in range(len(link_indices)):
            link_name = self._hand_link_names[link_indices[i]]
            T = self._dict_poses[link_name]
            J_link = self._dict_Jacobians[link_name]
            J_pt = get_point_Jacobian(
                T[:3, :3],
                J_link[3:, :],
                J_link[:3, :],
                points_link[i],
            )
            tau += J_pt.T @ forces_world[i]
        return tau

    def _solve_contact_force_qp(
        self, c_link_indices, c_points_cad, c_normals_cad, c_points_link
    ):
        n_contacts = len(c_link_indices)

        J_contact = np.zeros((n_contacts, 3, N_HAND_JOINTS))
        for i in range(n_contacts):
            link_name = self._hand_link_names[c_link_indices[i]]
            T = self._dict_poses[link_name]
            J_link = self._dict_Jacobians[link_name]
            J_contact[i] = get_point_Jacobian(
                T[:3, :3],
                J_link[3:, :],
                J_link[:3, :],
                c_points_link[i],
            )

        n = c_normals_cad[np.newaxis, :, :]
        c = c_points_cad[np.newaxis, :, :]
        Jr = J_contact[np.newaxis, :, :, :]

        Jo, Jr_contact, n, t1, t2 = compute_contact_frame(n, c, Jr)
        Jo = Jo[0]
        Jr_contact = Jr_contact[0]
        n = n[0]
        t1 = t1[0]
        t2 = t2[0]

        torque_limit = np.array(
            self._gen_config.get("torque_limit", [0.2] * N_HAND_JOINTS)
        )
        min_fn = self._gen_config.get("min_fn", 10.0)
        mu = self._gen_config.get("mu", 0.3)
        null_space_margin = self._gen_config.get("null_space_margin", 0.01)
        enable_torque_limit = self._gen_config.get("enable_torque_limit", True)
        enable_achievable_force = self._gen_config.get(
            "enable_achievable_force", True
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
        )

        if (
            solved
            and contact_epsilon is not None
            and self._gen_config.get("debug_eps_log", False)
        ):
            self._eps_log_counter = getattr(self, "_eps_log_counter", 0) + 1
            log_every = int(self._gen_config.get("debug_eps_log_every", 200))
            if self._eps_log_counter % log_every == 0:
                f_mag = np.linalg.norm(f, axis=1)
                if np.std(contact_epsilon) > 1e-9 and np.std(f_mag) > 1e-9:
                    corr = float(np.corrcoef(contact_epsilon, f_mag)[0, 1])
                else:
                    corr = float("nan")
                med = float(np.median(contact_epsilon))
                hi = contact_epsilon > med
                lo = ~hi
                f_hi = float(f_mag[hi].mean()) if hi.any() else 0.0
                f_lo = float(f_mag[lo].mean()) if lo.any() else 0.0
                ratio_hi_lo = f_hi / max(f_lo, 1e-6)
                sum_fn = float(f[:, 0].sum())
                print(
                    f"[eps] σ_p={sigma_p * 1000:.1f}mm "
                    f"σ_θ={np.degrees(sigma_th):.1f}° "
                    f"λ={eps_weight:.1e} P={len(contact_epsilon)}  "
                    f"ε[mm]:{contact_epsilon.min() * 1000:.2f}"
                    f"~{contact_epsilon.max() * 1000:.2f}  "
                    f"|f|[N]:{f_mag.min():.3f}~{f_mag.max():.3f}  "
                    f"corr(ε,|f|)={corr:+.2f}  "
                    f"|f|_hiε/|f|_loε={ratio_hi_lo:.2f}  "
                    f"sum_fn={sum_fn:.2f}/{min_fn}"
                )

        if solved:
            f_cad = f[:, 0:1] * n + f[:, 1:2] * t1 + f[:, 2:3] * t2
            f_world = self._convert_f_in_cad_to_world_frame(
                f_cad, self._cad_in_world_frame
            )
        else:
            f_cad = np.zeros((n_contacts, 3))
            f_world = np.zeros((n_contacts, 3))

        return J_contact, f_cad, f_world, solved

    def _compute_hand_pts_world(self, c_link_indices, c_points_link):
        n = len(c_link_indices)
        pts = np.zeros((n, 3))
        for i in range(n):
            T = self._dict_poses[self._hand_link_names[c_link_indices[i]]]
            pts[i] = T[:3, :3] @ c_points_link[i] + T[:3, 3]
        return pts

    def _compute_qp_force_closure(
        self, c_link_indices, c_points_cad, c_normals_cad, c_points_link
    ):
        J_contact, f_cad, f_world, solved = self._solve_contact_force_qp(
            c_link_indices, c_points_cad, c_normals_cad, c_points_link
        )

        if solved:
            self._f_in_cad_frame_vis = f_cad
            self._contact_points_in_cad_frame_vis = c_points_cad
            self._contact_points_world_vis = self._compute_hand_pts_world(
                c_link_indices, c_points_link
            )
            self._f_world_vis = f_world
            tau_qp = self._compute_tau_from_J_and_f(J_contact, f_world)
            self._prev_tau_qp = tau_qp
        else:
            tau_qp = self._prev_tau_qp

        return tau_qp

    def _compute_qp_impedance_torque(
        self, c_link_indices, c_points_cad, c_normals_cad, c_points_link
    ):
        J_contact, f_cad, f_world, solved = self._solve_contact_force_qp(
            c_link_indices, c_points_cad, c_normals_cad, c_points_link
        )

        if solved:
            k = self._k_for_contact_force_control
            target_cp_cad = c_points_cad + f_cad / k
            target_cp_world = self._tf_A2B(target_cp_cad, self._cad_in_world_frame)
            current_cp_world = self._compute_hand_pts_world(
                c_link_indices, c_points_link
            )
            control_f_world = k * (target_cp_world - current_cp_world)

            self._f_in_cad_frame_vis = f_cad
            self._contact_points_in_cad_frame_vis = c_points_cad
            self._contact_points_world_vis = current_cp_world
            self._target_cp_in_world_frame_vis = target_cp_world
            self._current_cp_in_world_frame_vis = current_cp_world
            self._f_world_vis = control_f_world

            tau = self._compute_tau_from_J_and_f(J_contact, control_f_world)
            self._prev_tau_qp = tau
        else:
            tau = self._prev_tau_qp

        return tau
