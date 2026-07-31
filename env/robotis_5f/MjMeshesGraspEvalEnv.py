"""Hand-only MuJoCo grasp-eval env (physics + LCM; control is external).

Ports the scene / object / grasp-pose side of ReactiveGrasp
``MjMeshesGraspEvalEnv`` for the RoPhiTutorial three-process pattern.
Control torque comes from ``MjEvalGraspControl`` over LCM — this env does
not compute feedforward / QP torques.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np
import trimesh
from omegaconf import DictConfig

from data_type.basic_types.NamedVecListData import NamedVecListData
from env.MujocoBaseEnv import MujocoBaseEnv
from utils.grasping.get_grasp_data import get_grasp_data
from utils.lie.se3 import invSE3
from utils.mode import VisMode
from utils.mujoco.rotation import quat2rotmat, rotmat2quat
from utils.random.random_generator import random_name

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


class MjMeshesGraspEvalEnv(MujocoBaseEnv):
    """Physics plant for mesh grasp evaluation (Robotis RH-5, freejoint object)."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        self._config = config
        scene = config.get("scene_params", {}) or {}
        self._scene_params_config = scene
        self._workspace_2d = scene["workspace_2d"]
        self._xmin, self._xmax = self._workspace_2d[0]
        self._ymin, self._ymax = self._workspace_2d[1]
        self._min_n_objs = int(scene.get("min_n_objects", 1))
        self._max_n_objs = int(scene.get("max_n_objects", 1))
        self._table_height = 0.05

        self._obj_path = _resolve_path(config["obj_path"])
        self._object_density = float(config.get("object_density", 500))
        self._texture_img_size = list(config.get("texture_img_size", [512, 512]))
        self._obj_mesh2init_quat = list(
            config.get("obj_mesh2init_quat", [1.0, 0.0, 0.0, 0.0])
        )
        self._obj_name = config.get("obj_name", None)
        self._mesh_name = config.get("mesh_name", None)

        self.dict_mj_objects = {}
        self.dict_mj_obstacles = {}
        self.dict_object_point_cloud_mesh_frame = {}

        # Perturbation
        perturb_cfg = config.get("perturbation", {}) or {}
        self._perturbation_active = False
        self._perturbation_step_count = 0
        self._perturbation_T_target = None
        self._perturbation_body_id = None
        self._perturbation_Kp = 0.0
        self._perturbation_Kr = 0.0
        self._perturbation_Dp = 0.0
        self._perturbation_Dr = 0.0
        self._perturbation_apply_steps = int(perturb_cfg.get("apply_steps", 500))
        self._perturbation_sigma_pos = float(perturb_cfg.get("sigma_pos", 0.0))
        self._perturbation_sigma_rot = float(perturb_cfg.get("sigma_rot", 0.3))
        self._perturbation_Kp_scale = float(perturb_cfg.get("Kp_scale", 500.0))
        self._perturbation_Kr_scale = float(perturb_cfg.get("Kr_scale", 500.0))
        self._perturbation_max_force = perturb_cfg.get("max_force", None)
        self._perturbation_max_torque = perturb_cfg.get("max_torque", None)
        self._perturbation_rng = np.random.RandomState(seed=42)
        self._perturbation_force_world_vis = None
        self._perturbation_point_world_vis = None

        # LCM channels
        pub = config["pub_manager"]
        self._obj_pose_bb2world_channel = pub["named_vec_list_channel"]
        self._contact_forces_channel = pub.get("named_vec_list_channel_2", None)
        self._obj_pose_bb2world_data = None
        self._last_obj_pose_bb2world_update_time = 0.0
        self._obj_pose_bb2world_update_dt = 1.0 / 60.0
        self._last_contact_forces_update_time = 0.0
        self._contact_forces_update_dt = 1.0 / 60.0

        # Latest controller contact forces for MuJoCo overlay:
        # (P, 6) = [px, py, pz, fx, fy, fz] in world, excluding "perturbation".
        self._ctrl_contact_forces_world = np.zeros((0, 6))
        # Optional normals: (P, 6) = [px, py, pz, nx, ny, nz]
        self._ctrl_contact_normals_world = np.zeros((0, 6))
        self.extr_sub_que_dict = None

        # Grasp state (filled in initialize)
        self._grasp_data = None
        self._grasp_idx = 0
        self._n_grasps = 0
        self._mesh = None
        self._bb_to_cad = np.eye(4)
        self._cad_to_bb = np.eye(4)
        self._obj_center_in_cad_frame = np.zeros(3)

        super().__init__(config, *args, **kwargs)
        self._reset_object_pose()

    def set_sub_que_dict(self, que_dict: dict) -> None:
        """Also attach extrinsic NamedVecList queues (controller contact forces)."""
        super().set_sub_que_dict(que_dict)
        self.extr_sub_que_dict = que_dict.get("extr_sub_que_dict", {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        model = self._config.get("model", "robotis_5f")
        grasp_predictions_path = os.path.join(self._obj_path, "grasp_predictions")
        self._grasp_data = get_grasp_data(model, grasp_predictions_path)
        self._grasp_idx = int(self._config.get("grasp_idx", 0))
        self._n_grasps = len(self._grasp_data.get_grasp_poses_in_cad_frame())

        mesh = trimesh.load(os.path.join(self._obj_path, "textured_mesh.obj"))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        self._mesh = mesh
        self._obj_center_in_cad_frame = np.asarray(self._mesh.centroid)

        # Gravity off by default (matches mujoco_eval_grasp.py)
        if bool(self._config.get("gravity_off_on_start", True)):
            self.mj_model.opt.gravity[:] = 0.0
            print("[Env] Gravity OFF at start (press 'g' to toggle).")

        self._set_object_and_grasp_pose()
        print(
            f"[Env] Grasp eval ready: object={Path(self._obj_path).name}, "
            f"grasps={self._n_grasps}. Keys: r=reset, n=next, p=perturb, g=gravity."
        )

    def _init_viewer(self) -> None:
        super()._init_viewer()
        with self.mj_viewer.lock():
            cam_cfg = self._config.get("viewer_cam", {}) or {}
            if cam_cfg:
                self.mj_viewer.cam.distance = cam_cfg.get("distance", 0.6)
                self.mj_viewer.cam.elevation = cam_cfg.get("elevation", -30)
                self.mj_viewer.cam.azimuth = cam_cfg.get("azimuth", 225)
                lookat = cam_cfg.get("lookat", [0.0, 0.0, 0.1])
                self.mj_viewer.cam.lookat = np.array(lookat)

            vis_cfg = self._config.get("viewer_vis", {}) or {}
            flag_map = {
                "contact_point": mj.mjtVisFlag.mjVIS_CONTACTPOINT,
                "contact_force": mj.mjtVisFlag.mjVIS_CONTACTFORCE,
                "contact_split": mj.mjtVisFlag.mjVIS_CONTACTSPLIT,
            }
            for key, flag in flag_map.items():
                if key in vis_cfg:
                    self.mj_viewer.opt.flags[flag] = vis_cfg[key]

    def reset(self) -> None:
        self._pause = False
        self._finished = False
        import time as _time

        self._last_real_t = _time.time()
        self._last_view_t = _time.time()

        self._modify_mjspec()
        self.mj_model = self.mj_spec.compile()
        self.mj_data = mj.MjData(self.mj_model)
        self.platform.set_mj_data_name_idx(self.mj_data)
        self.mj_model.opt.timestep = self._sim_dt
        self._qd_prev = self.mj_data.qvel.copy()

        self.platform.reset(self.mj_data)
        mj.mj_forward(self.mj_model, self.mj_data)
        self._reset_object_pose()

        if self.vis_mode == VisMode.ON and self.mj_viewer is not None:
            self.mj_viewer.close()
            self._init_viewer()

        if self._grasp_data is not None:
            self._set_object_and_grasp_pose()
        print("[Env] Environment reset complete.")

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _handle_key(self, key: str) -> None:
        if key == "g":
            if np.linalg.norm(self.mj_model.opt.gravity) > 0:
                self._saved_gravity = self.mj_model.opt.gravity.copy()
                self.mj_model.opt.gravity[:] = 0
                print("[Env] Gravity OFF")
            else:
                grav = getattr(self, "_saved_gravity", np.array([0.0, 0.0, -9.81]))
                self.mj_model.opt.gravity[:] = grav
                print(f"[Env] Gravity ON ({grav})")
        elif key == "p":
            self._start_perturbation()
        elif key == "r":
            self._stop_perturbation()
            print("[Env] Resetting object and grasp pose.")
            self._set_object_and_grasp_pose()
        elif key == "n":
            self._stop_perturbation()
            if self._n_grasps <= 0:
                return
            self._grasp_idx = (self._grasp_idx + 1) % self._n_grasps
            print(f"[Env] Grasp {self._grasp_idx + 1}/{self._n_grasps} selected.")
            self._set_object_and_grasp_pose()
        else:
            super()._handle_key(key)

    def _key_callback(self, keycode) -> None:
        try:
            ch = chr(keycode)
        except (ValueError, OverflowError):
            ch = ""
        if ch in ("g", "p", "r", "n"):
            self._handle_key(ch)
            return
        super()._key_callback(keycode)

    # ------------------------------------------------------------------
    # Control / step (external tau only)
    # ------------------------------------------------------------------

    def _apply_sim_control(self, ctrl_data) -> None:
        if ctrl_data is not None:
            self.platform.apply_sim_control(ctrl_data, self.mj_data)

    def _step_simulation(self) -> None:
        self._drain_controller_contact_forces()
        if self._perturbation_active:
            self._step_perturbation()
        super()._step_simulation()

    def _drain_controller_contact_forces(self) -> None:
        """Pull latest QP contact points/forces from the controller LCM channel."""
        if not self.extr_sub_que_dict:
            return
        ch = self._contact_forces_channel
        if ch is None or ch not in self.extr_sub_que_dict:
            return
        que = self.extr_sub_que_dict[ch]
        latest = None
        while not que.empty():
            latest = que.get_nowait()
        if latest is None:
            return
        _, name_list, vec_list = latest.get_data()
        vec_list = np.asarray(vec_list, dtype=np.float64)
        if vec_list.ndim != 2 or vec_list.shape[1] < 6:
            return

        force_rows = []
        normal_rows = []
        for name, vec in zip(name_list, vec_list):
            name = str(name)
            if name == "perturbation" or name == "empty" or name == "net":
                continue
            if name.startswith("n_"):
                normal_rows.append(vec[:6])
            else:
                force_rows.append(vec[:6])
        self._ctrl_contact_forces_world = (
            np.asarray(force_rows, dtype=np.float64)
            if force_rows
            else np.zeros((0, 6))
        )
        self._ctrl_contact_normals_world = (
            np.asarray(normal_rows, dtype=np.float64)
            if normal_rows
            else np.zeros((0, 6))
        )

    def _update_vis(self) -> None:
        if (
            self.vis_mode == VisMode.ON
            and self.mj_viewer is not None
            and self.mj_viewer.is_running()
        ):
            self._draw_contact_overlay_in_mj_viewer()
        super()._update_vis()

    def _rotation_from_z_to_dir(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return np.eye(3)
        d = d / n
        if abs(d[2]) < 0.9999:
            x = np.cross(d, np.array([0.0, 0.0, 1.0]))
        else:
            x = np.cross(d, np.array([1.0, 0.0, 0.0]))
        x = x / np.linalg.norm(x)
        y = np.cross(d, x)
        y = y / np.linalg.norm(y)
        return np.column_stack([x, y, d])

    def _draw_contact_overlay_in_mj_viewer(self) -> None:
        """Draw QP contact spheres + force arrows (+ optional normals) in user_scn."""
        if not hasattr(self.mj_viewer, "user_scn"):
            return
        user_scn = self.mj_viewer.user_scn
        n_geom = 0
        eye3 = np.eye(3).flatten().astype(np.float64).reshape(9, 1)

        # Contact points (QP) — bright red spheres
        pts_f = self._ctrl_contact_forces_world
        radius = 0.004
        rgba_pt = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for i in range(pts_f.shape[0]):
            if n_geom >= user_scn.maxgeom:
                break
            geom = user_scn.geoms[n_geom]
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=np.array([radius, 0, 0], dtype=np.float64).reshape(3, 1),
                pos=pts_f[i, :3].astype(np.float64).reshape(3, 1),
                mat=eye3,
                rgba=rgba_pt.reshape(4, 1),
            )
            n_geom += 1

        # Force arrows — magenta boxes
        force_scale = 0.05
        min_length = 0.005
        max_length = 0.25
        thickness = 0.0025
        rgba_f = np.array([1.0, 0.0, 0.8, 1.0], dtype=np.float32)
        for i in range(pts_f.shape[0]):
            if n_geom >= user_scn.maxgeom:
                break
            p = pts_f[i, :3]
            f = pts_f[i, 3:6]
            mag = np.linalg.norm(f)
            if mag < 1e-9:
                continue
            length = float(np.clip(force_scale * mag, min_length, max_length))
            tip = p + (f / mag) * length
            pos = (p + tip) / 2
            R = self._rotation_from_z_to_dir(tip - p)
            geom = user_scn.geoms[n_geom]
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_BOX,
                size=np.array(
                    [thickness / 2, thickness / 2, length / 2], dtype=np.float64
                ).reshape(3, 1),
                pos=pos.astype(np.float64).reshape(3, 1),
                mat=R.flatten().astype(np.float64).reshape(9, 1),
                rgba=rgba_f.reshape(4, 1),
            )
            n_geom += 1

        # Contact normals — green short sticks
        pts_n = self._ctrl_contact_normals_world
        normal_length = 0.015
        rgba_n = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        for i in range(pts_n.shape[0]):
            if n_geom >= user_scn.maxgeom:
                break
            p = pts_n[i, :3]
            d = pts_n[i, 3:6]
            dn = np.linalg.norm(d)
            if dn < 1e-9:
                continue
            d = d / dn
            tip = p + normal_length * d
            pos = (p + tip) / 2
            R = self._rotation_from_z_to_dir(d)
            geom = user_scn.geoms[n_geom]
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_BOX,
                size=np.array(
                    [0.001, 0.001, normal_length / 2], dtype=np.float64
                ).reshape(3, 1),
                pos=pos.astype(np.float64).reshape(3, 1),
                mat=R.flatten().astype(np.float64).reshape(9, 1),
                rgba=rgba_n.reshape(4, 1),
            )
            n_geom += 1

        # Perturbation force (cyan)
        if self._perturbation_force_world_vis is not None:
            p = self._perturbation_point_world_vis
            f = self._perturbation_force_world_vis
            mag = np.linalg.norm(f)
            if mag > 1e-9 and n_geom < user_scn.maxgeom:
                length = float(np.clip(force_scale * mag, min_length, max_length))
                tip = p + (f / mag) * length
                pos = (p + tip) / 2
                R = self._rotation_from_z_to_dir(tip - p)
                rgba_p = np.array([0.0, 0.9, 1.0, 1.0], dtype=np.float32)
                geom = user_scn.geoms[n_geom]
                mj.mjv_initGeom(
                    geom,
                    type=mj.mjtGeom.mjGEOM_BOX,
                    size=np.array(
                        [0.003, 0.003, length / 2], dtype=np.float64
                    ).reshape(3, 1),
                    pos=pos.astype(np.float64).reshape(3, 1),
                    mat=R.flatten().astype(np.float64).reshape(9, 1),
                    rgba=rgba_p.reshape(4, 1),
                )
                n_geom += 1

        user_scn.ngeom = n_geom

    # ------------------------------------------------------------------
    # Grasp / object pose
    # ------------------------------------------------------------------

    def _set_object_and_grasp_pose(self) -> None:
        if self._grasp_data is None:
            return
        grasp_pose = self._grasp_data.get_grasp_poses_in_cad_frame()[self._grasp_idx]
        q_gripper = grasp_pose[:20]
        SE3_grasp_pose_in_cad = np.eye(4)
        SE3_grasp_pose_in_cad[:3, 3] = grasp_pose[20:23]
        SE3_grasp_pose_in_cad[:3, :3] = grasp_pose[23:].reshape(3, 3)

        SE3_grasp_in_palm = self._grasp_data.get_SE3_grasp_pose_in_palm_frame()
        target_cad_in_palm = SE3_grasp_in_palm @ invSE3(SE3_grasp_pose_in_cad)
        palm_in_world = self._get_palm_pose_in_world_frame()
        target_cad_in_world = palm_in_world @ target_cad_in_palm

        self._set_object_pose(target_cad_in_world)
        self._set_joint_position(q_gripper)
        mj.mj_forward(self.mj_model, self.mj_data)

    def _set_joint_position(self, q_gripper) -> None:
        self.mj_data.qpos[: len(q_gripper)] = q_gripper
        self.mj_data.qvel[: len(q_gripper)] = 0.0

    def _set_object_pose(self, cad_in_world) -> None:
        pos = cad_in_world[:3, 3]
        quat = rotmat2quat(cad_in_world[:3, :3])
        for obj_name in self.dict_mj_objects:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, obj_name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            self.mj_data.qpos[qpos_adr : qpos_adr + 3] = pos
            self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
            self.mj_data.qvel[qpos_adr : qpos_adr + 6] = 0.0

    def _get_palm_pose_in_world_frame(self) -> np.ndarray:
        mj.mj_forward(self.mj_model, self.mj_data)
        body_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, "palm")
        palm = np.eye(4)
        palm[:3, :3] = self.mj_data.xmat[body_id].reshape(3, 3).copy()
        palm[:3, 3] = self.mj_data.xpos[body_id].copy()
        return palm

    def _reset_object_pose(self) -> None:
        for obj_name in self.dict_mj_objects:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, obj_name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            x = np.random.uniform(self._xmin, self._xmax)
            y = np.random.uniform(self._ymin, self._ymax)
            self.mj_data.qpos[qpos_adr : qpos_adr + 3] = [
                x,
                y,
                0.3 + self._table_height,
            ]
            self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = self._obj_mesh2init_quat
            self.mj_data.qvel[qpos_adr : qpos_adr + 6] = 0.0
            mj.mj_forward(self.mj_model, self.mj_data)

    # ------------------------------------------------------------------
    # Perturbation (from MjMeshesGraspEvalEnv)
    # ------------------------------------------------------------------

    def _start_perturbation(self) -> None:
        if not self.dict_mj_objects:
            return
        obj_name = list(self.dict_mj_objects.keys())[0]
        body_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, obj_name)
        T_settled = np.eye(4)
        T_settled[:3, :3] = self.mj_data.xmat[body_id].reshape(3, 3)
        T_settled[:3, 3] = self.mj_data.xpos[body_id]

        center = T_settled[:3, :3] @ self._obj_center_in_cad_frame + T_settled[:3, 3]
        # Random SO(3) perturbation about center
        axis = self._perturbation_rng.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = self._perturbation_rng.normal(scale=self._perturbation_sigma_rot)
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        T_target = np.eye(4)
        T_target[:3, :3] = R @ T_settled[:3, :3]
        T_target[:3, 3] = R @ (T_settled[:3, 3] - center) + center
        if self._perturbation_sigma_pos > 0:
            T_target[:3, 3] += self._perturbation_rng.normal(
                scale=self._perturbation_sigma_pos, size=3
            )

        mass = self.mj_model.body_mass[body_id]
        inertia_max = np.max(self.mj_model.body_inertia[body_id])
        Kp = self._perturbation_Kp_scale * mass
        Kr = self._perturbation_Kr_scale * inertia_max
        self._perturbation_active = True
        self._perturbation_step_count = 0
        self._perturbation_T_target = T_target
        self._perturbation_body_id = body_id
        self._perturbation_Kp = Kp
        self._perturbation_Kr = Kr
        self._perturbation_Dp = 2.0 * np.sqrt(Kp * mass)
        self._perturbation_Dr = 2.0 * np.sqrt(Kr * inertia_max)
        print(
            f"[Env] Perturbation started: steps={self._perturbation_apply_steps}, "
            f"Kp={Kp:.1f}, Kr={Kr:.4f}"
        )

    def _stop_perturbation(self) -> None:
        if self._perturbation_body_id is not None:
            self.mj_data.xfrc_applied[self._perturbation_body_id] = 0.0
        self._perturbation_active = False
        self._perturbation_T_target = None
        self._perturbation_force_world_vis = None
        self._perturbation_point_world_vis = None

    def _step_perturbation(self) -> None:
        bid = self._perturbation_body_id
        T_cur = np.eye(4)
        T_cur[:3, :3] = self.mj_data.xmat[bid].reshape(3, 3)
        T_cur[:3, 3] = self.mj_data.xpos[bid]
        vel = np.zeros(6)
        mj.mj_objectVelocity(
            self.mj_model, self.mj_data, mj.mjtObj.mjOBJ_BODY, bid, vel, 0
        )

        # Attractive wrench toward target (same structure as generate_attract_wrench)
        R_err = self._perturbation_T_target[:3, :3] @ T_cur[:3, :3].T
        skew = 0.5 * (R_err - R_err.T)
        omega_err = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        pos_err = self._perturbation_T_target[:3, 3] - T_cur[:3, 3]
        force = self._perturbation_Kp * pos_err - self._perturbation_Dp * vel[3:]
        torque = self._perturbation_Kr * omega_err - self._perturbation_Dr * vel[:3]

        if self._perturbation_max_force is not None:
            fn = np.linalg.norm(force)
            if fn > self._perturbation_max_force:
                force *= self._perturbation_max_force / fn
        if self._perturbation_max_torque is not None:
            tn = np.linalg.norm(torque)
            if tn > self._perturbation_max_torque:
                torque *= self._perturbation_max_torque / tn

        self.mj_data.xfrc_applied[bid, :3] = force
        self.mj_data.xfrc_applied[bid, 3:] = torque
        self._perturbation_force_world_vis = force
        self._perturbation_point_world_vis = T_cur[:3, 3]
        self._perturbation_step_count += 1

        pos_e = np.linalg.norm(pos_err)
        rot_e = np.linalg.norm(omega_err)
        if (
            (pos_e < 0.002 and rot_e < 0.05)
            or self._perturbation_step_count >= self._perturbation_apply_steps
        ):
            print(
                f"[Env] Perturbation ended after {self._perturbation_step_count} steps "
                f"(pos_err={pos_e*1000:.1f}mm)"
            )
            self._stop_perturbation()

    # ------------------------------------------------------------------
    # Scene / mesh object
    # ------------------------------------------------------------------

    def _modify_mjspec(self) -> None:
        self._remove_all_objects_from_mj_spec()
        n = int(np.random.randint(self._min_n_objs, self._max_n_objs + 1))
        for i in range(n):
            self._add_random_object_to_mj_spec(self.mj_spec.worldbody, f"object_{i}")

    def _remove_all_objects_from_mj_spec(self) -> None:
        self.dict_object_point_cloud_mesh_frame = {}
        for obj_body in list(self.dict_mj_objects.values()):
            self.mj_spec.detach_body(obj_body)
        self.dict_mj_objects = {}

    def _get_obj(self):
        obj_files = [f for f in os.listdir(self._obj_path) if f.endswith(".obj")]
        if not obj_files:
            raise ValueError(f"No .obj in {self._obj_path}")
        obj_file_path = Path(self._obj_path) / obj_files[0]
        obj_stem = obj_file_path.stem

        trimesh_mesh = trimesh.load(str(obj_file_path))
        if isinstance(trimesh_mesh, trimesh.Scene):
            trimesh_mesh = trimesh_mesh.dump(concatenate=True)
        self._cad_to_bb, _ = trimesh.bounds.oriented_bounds(trimesh_mesh)
        self._bb_to_cad = np.linalg.inv(self._cad_to_bb)

        with open(str(obj_file_path), "rb") as f:
            obj_string_visual = f.read()

        obj_name = self._mesh_name or Path(self._obj_path).stem

        cache_root = Path(self._obj_path) / "convexification" / obj_stem
        cached_parts = sorted(cache_root.glob("part_*.obj"))
        if not cached_parts:
            raise FileNotFoundError(
                f"No convexification cache at {cache_root}. "
                "Generate parts offline (CoACD) or copy from ReactiveGrasp."
            )
        obj_list_strings_collision = []
        for p in cached_parts:
            with open(p, "rb") as f:
                obj_list_strings_collision.append(f.read())
        return obj_name, obj_string_visual, obj_list_strings_collision

    def _get_texture(self) -> bytes:
        for file in os.listdir(self._obj_path):
            if file.endswith(".png"):
                img = cv2.imread(os.path.join(self._obj_path, file), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img,
                    (self._texture_img_size[0], self._texture_img_size[1]),
                    interpolation=cv2.INTER_LINEAR,
                )
                return img.astype(np.uint8).tobytes()
        w, h = self._texture_img_size
        return np.full((h, w, 3), 180, dtype=np.uint8).tobytes()

    def _add_random_object_to_mj_spec(self, worldbody, object_name=None) -> None:
        obj_name, obj_string_visual, obj_list_strings_collision = self._get_obj()
        texture_bytes = self._get_texture()

        if not hasattr(self, "_mesh_tmp_dir"):
            self._mesh_tmp_dir = tempfile.mkdtemp(prefix="mj_mesh_")
        tmp_dir = self._mesh_tmp_dir

        rand_vis = random_name()
        rand_tex = random_name()
        rand_mat = random_name()

        list_meshes = []
        mesh_vis = self.mj_spec.add_mesh()
        mesh_vis.name = rand_vis + "_visual"
        vis_filename = f"{rand_vis}_visual.obj"
        mesh_vis.file = os.path.join(tmp_dir, vis_filename)
        with open(mesh_vis.file, "wb") as f:
            f.write(obj_string_visual)
        list_meshes.append(mesh_vis)

        for j, col_str in enumerate(obj_list_strings_collision):
            mesh_col = self.mj_spec.add_mesh()
            mesh_col.name = f"{rand_vis}_collision_{j}"
            col_filename = f"{rand_vis}_collision_{j}.obj"
            mesh_col.file = os.path.join(tmp_dir, col_filename)
            with open(mesh_col.file, "wb") as f:
                data = col_str if isinstance(col_str, bytes) else col_str.encode()
                f.write(data)
            list_meshes.append(mesh_col)

        texture = self.mj_spec.add_texture(
            name=rand_tex,
            type=mj.mjtTexture.mjTEXTURE_2D,
            width=self._texture_img_size[0],
            height=self._texture_img_size[1],
        )
        texture.data = texture_bytes
        material = self.mj_spec.add_material(name=rand_mat)
        material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = rand_tex

        x = np.random.uniform(self._xmin, self._xmax)
        y = np.random.uniform(self._ymin, self._ymax)
        body_name = obj_name if object_name is None else obj_name
        self.dict_mj_objects[body_name] = worldbody.add_body(
            name=body_name,
            pos=[x, y, 0.3 + self._table_height],
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.dict_mj_objects[body_name].add_joint(
            name=body_name,
            type=mj.mjtJoint.mjJNT_FREE,
            axis=[1, 0, 0],
            pos=[0, 0, 0],
        )
        for mesh in list_meshes:
            if "collision" in mesh.name:
                self.dict_mj_objects[body_name].add_geom(
                    name=mesh.name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    density=self._object_density,
                    contype=1,
                    conaffinity=1,
                    condim=6,
                    friction=[0.05, 0.005, 0.0001],
                    rgba=[0, 0, 0, 0],
                )
            else:
                self.dict_mj_objects[body_name].add_geom(
                    name=mesh.name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    material=rand_mat,
                    contype=0,
                    conaffinity=0,
                    density=0,
                )

    # ------------------------------------------------------------------
    # LCM publish
    # ------------------------------------------------------------------

    def _get_object_pose_bb2world(self) -> np.ndarray:
        list_poses = []
        for obj_name in self.dict_mj_objects:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, obj_name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            pos = self.mj_data.qpos[qpos_adr : qpos_adr + 3]
            quat = self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7]
            rot = quat2rotmat(quat)
            T_cad2world = np.eye(4)
            T_cad2world[:3, :3] = rot
            T_cad2world[:3, 3] = pos
            T_bb2world = T_cad2world @ self._bb_to_cad
            list_poses.append(
                np.concatenate([T_bb2world[:3, 3], T_bb2world[:3, :3].flatten()])
            )
        return np.array(list_poses)

    def _sync_extr_data_from_sim(self, extr_pub_que_dict) -> None:
        t = self.mj_data.time
        if (
            self._obj_pose_bb2world_channel in extr_pub_que_dict
            and t - self._last_obj_pose_bb2world_update_time
            > self._obj_pose_bb2world_update_dt
        ):
            self._last_obj_pose_bb2world_update_time = t
            true_poses = self._get_object_pose_bb2world()
            name_list = list(self.dict_mj_objects.keys())
            vec_list = true_poses

            obj_name = name_list[0]
            bid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, obj_name)
            physics = np.zeros(12)
            physics[0] = self.mj_model.body_mass[bid]
            physics[1:4] = self.mj_model.body_ipos[bid]
            physics[4:7] = self.mj_model.opt.gravity
            name_list = name_list + ["physics_params"]
            vec_list = np.concatenate([vec_list, physics[np.newaxis, :]], axis=0)

            data = NamedVecListData(
                name=self._obj_pose_bb2world_channel,
                num_vecs=len(name_list),
                vec_dim=12,
            )
            data.set_time(t)
            data.set_data(t=t, name_list=name_list, vec_list=vec_list)
            extr_pub_que_dict[self._obj_pose_bb2world_channel].put(copy.deepcopy(data))

        # Perturbation is drawn locally in MuJoCo; Viser gets contact forces
        # from the controller. Optionally append perturbation onto the same
        # LCM channel without wiping controller contacts (name="perturbation").
        if (
            self._contact_forces_channel is not None
            and self._contact_forces_channel in extr_pub_que_dict
            and self._perturbation_force_world_vis is not None
            and t - self._last_contact_forces_update_time
            > self._contact_forces_update_dt
        ):
            self._last_contact_forces_update_time = t
            # Merge latest controller contacts (if any) + perturbation row so
            # a Viser client that only sees this packet still gets both.
            name_list = []
            vecs_list = []
            for i in range(self._ctrl_contact_forces_world.shape[0]):
                name_list.append(str(i))
                vecs_list.append(self._ctrl_contact_forces_world[i])
            for i in range(self._ctrl_contact_normals_world.shape[0]):
                name_list.append(f"n_{i}")
                vecs_list.append(self._ctrl_contact_normals_world[i])
            name_list.append("perturbation")
            vecs_list.append(
                np.concatenate(
                    [
                        self._perturbation_point_world_vis,
                        self._perturbation_force_world_vis,
                    ]
                )
            )
            vecs = np.asarray(vecs_list, dtype=np.float64)
            data = NamedVecListData(
                name=self._contact_forces_channel,
                num_vecs=len(name_list),
                vec_dim=6,
            )
            data.set_time(t)
            data.set_data(t=t, name_list=name_list, vec_list=vecs)
            extr_pub_que_dict[self._contact_forces_channel].put(copy.deepcopy(data))
