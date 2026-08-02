"""Robot-free MuJoCo env: table + textured mesh + RGB-D camera over LCM."""

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

from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.NamedVecListData import NamedVecListData
from data_type.basic_types.RGBDData import RGBDData
from env.MujocoBaseEnv import MujocoBaseEnv
from utils.communication.lcm.image_conversion import DTYPE_TO_CHANNEL_TYPE
from utils.mode import VisMode
from utils.mujoco.camera import (
    get_camera_extrinsics,
    get_camera_intrinsics,
    initialize_camera_renderer,
    render_image_from_camera,
)
from utils.mujoco.rotation import quat2rotmat
from utils.random.random_generator import random_name

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


class TableMeshCamEnv(MujocoBaseEnv):
    """Table + freejoint textured object; publishes RGBD and GT bb2world pose."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        self._config = config
        self._list_camera_names = list(config["sensor"]["extrinsic"].keys())

        self._obj_path = _resolve_path(config["obj_path"])
        self._object_density = float(config.get("object_density", 500.0))
        self._texture_img_size = list(config.get("texture_img_size", [512, 512]))
        self._obj_mesh2init_quat = list(
            config.get("obj_mesh2init_quat", [1.0, 0.0, 0.0, 0.0])
        )
        self._mesh_name = config.get("mesh_name", "predefined_obj")
        self._object_name = config.get("object_name", self._mesh_name)
        self._table_height = float(config.get("table_height", 0.42))
        self._obj_xy = list(config.get("obj_xy", [0.0, 0.0]))
        self._spin_yaw_rate = float(config.get("spin_yaw_rate", 0.4))
        self._spinning = bool(config.get("spin_on_start", False))

        pub = config["pub_manager"]
        self._obj_pose_channel = pub.get(
            "named_vec_list_channel", "sim_obj_pose_bb2world"
        )
        self._obj_pose_pub_freq = float(config.get("obj_pose_pub_freq", 30.0))
        self._last_obj_pose_update_time = 0.0
        self._obj_pose_update_dt = 1.0 / max(self._obj_pose_pub_freq, 1e-3)

        self.dict_mj_objects = {}
        self._cad_to_bb = np.eye(4)
        self._bb_to_cad = np.eye(4)
        self._mesh_tmp_dir = None

        # Camera bookkeeping filled after model load in __init__ of MujocoBaseEnv
        self._dict_camera_channels = {}
        self._dict_last_sensor_update_time = {}
        self._dict_sensor_update_dt = {}
        self._dict_cam_rgb_enable = {}
        self._dict_cam_depth_enable = {}
        self._dict_cam_data = {}
        self._dict_cam_info = {}
        self._dict_cam_renderer = {}

        super().__init__(config, *args, **kwargs)

        for cam_name in self._list_camera_names:
            self._dict_camera_channels[cam_name] = pub.get(
                f"{cam_name}_channel", cam_name
            )
            self._dict_last_sensor_update_time[cam_name] = self.mj_data.time
            self._dict_sensor_update_dt[cam_name] = (
                1.0 / config["sensor"]["extrinsic"][cam_name]["frequency"]
            )
            self._dict_cam_rgb_enable[cam_name] = config["sensor"]["extrinsic"][
                cam_name
            ]["rgb_enable"]
            self._dict_cam_depth_enable[cam_name] = config["sensor"]["extrinsic"][
                cam_name
            ]["depth_enable"]

        self._place_object()

    def initialize(self) -> None:
        print(
            f"[Env] TableMeshCamEnv ready: object={Path(self._obj_path).name}, "
            f"mesh_name={self._mesh_name}. Keys: s=spin, r=reset, q=quit."
        )
        if self._spinning:
            print(f"[Env] Spin ON (yaw rate={self._spin_yaw_rate:.2f} rad/s).")

    def _handle_key(self, key: str) -> None:
        if key == "s":
            self._spinning = not self._spinning
            print(f"[Env] Spin: {self._spinning}")
            if not self._spinning:
                self._zero_object_velocity()
            return
        super()._handle_key(key)

    def reset(self) -> None:
        self._pause = False
        self._finished = False
        import time as _time

        self._last_real_t = _time.time()
        self._last_view_t = _time.time()

        # Reload scene so mesh injection is clean.
        self.mj_spec = mj.MjSpec.from_file(self._scene_xml_path)
        self._modify_mjspec()
        self.mj_model = self.mj_spec.compile()
        self.mj_data = mj.MjData(self.mj_model)
        self.platform.set_mj_data_name_idx(self.mj_data)
        self.mj_model.opt.timestep = self._sim_dt
        self._qd_prev = self.mj_data.qvel.copy()

        self.platform.reset(self.mj_data)
        mj.mj_forward(self.mj_model, self.mj_data)
        self._initialize_sensors(self._sensor_config)
        self._place_object()

        for cam_name in self._list_camera_names:
            self._dict_last_sensor_update_time[cam_name] = self.mj_data.time
        self._last_obj_pose_update_time = self.mj_data.time

        if self.vis_mode == VisMode.ON and self.mj_viewer is not None:
            self.mj_viewer.close()
            self._init_viewer()

        print("[Env] Environment reset complete.")

    def _init_viewer(self) -> None:
        super()._init_viewer()
        with self.mj_viewer.lock():
            cam_cfg = self._config.get("viewer_cam", {}) or {}
            self.mj_viewer.cam.distance = cam_cfg.get("distance", 1.4)
            self.mj_viewer.cam.elevation = cam_cfg.get("elevation", -35)
            self.mj_viewer.cam.azimuth = cam_cfg.get("azimuth", 140)
            lookat = cam_cfg.get("lookat", [0.0, 0.0, 0.45])
            self.mj_viewer.cam.lookat = np.array(lookat)
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    # ------------------------------------------------------------------
    # Scene / mesh
    # ------------------------------------------------------------------

    def _modify_mjspec(self) -> None:
        self.dict_mj_objects = {}
        self._add_textured_object_to_mj_spec(self.mj_spec.worldbody)

    def _get_obj_bytes(self):
        obj_files = sorted(
            f for f in os.listdir(self._obj_path) if f.endswith(".obj")
        )
        if not obj_files:
            raise ValueError(f"No .obj in {self._obj_path}")
        # Prefer textured_mesh.obj when present.
        preferred = "textured_mesh.obj"
        obj_file = preferred if preferred in obj_files else obj_files[0]
        obj_file_path = Path(self._obj_path) / obj_file
        obj_stem = obj_file_path.stem

        trimesh_mesh = trimesh.load(str(obj_file_path), force="mesh")
        if isinstance(trimesh_mesh, trimesh.Scene):
            trimesh_mesh = trimesh_mesh.dump(concatenate=True)
        self._cad_to_bb, _ = trimesh.bounds.oriented_bounds(trimesh_mesh)
        self._bb_to_cad = np.linalg.inv(self._cad_to_bb)

        with open(str(obj_file_path), "rb") as f:
            obj_string_visual = f.read()

        cache_root = Path(self._obj_path) / "convexification" / obj_stem
        cached_parts = sorted(cache_root.glob("part_*.obj"))
        if cached_parts:
            obj_list_strings_collision = [p.read_bytes() for p in cached_parts]
        else:
            # Visual mesh only for collision if no convex parts.
            obj_list_strings_collision = [obj_string_visual]

        return self._mesh_name, obj_string_visual, obj_list_strings_collision

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

    def _add_textured_object_to_mj_spec(self, worldbody) -> None:
        obj_name, obj_string_visual, obj_list_strings_collision = self._get_obj_bytes()
        texture_bytes = self._get_texture()

        if self._mesh_tmp_dir is None:
            self._mesh_tmp_dir = tempfile.mkdtemp(prefix="table_mesh_cam_")
        tmp_dir = self._mesh_tmp_dir

        rand_vis = random_name()
        rand_tex = random_name()
        rand_mat = random_name()

        list_meshes = []
        mesh_vis = self.mj_spec.add_mesh()
        mesh_vis.name = rand_vis + "_visual"
        mesh_vis.file = os.path.join(tmp_dir, f"{rand_vis}_visual.obj")
        with open(mesh_vis.file, "wb") as f:
            f.write(obj_string_visual)
        list_meshes.append(mesh_vis)

        for j, col_str in enumerate(obj_list_strings_collision):
            mesh_col = self.mj_spec.add_mesh()
            mesh_col.name = f"{rand_vis}_collision_{j}"
            mesh_col.file = os.path.join(tmp_dir, f"{rand_vis}_collision_{j}.obj")
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

        body = worldbody.add_body(
            name=obj_name,
            pos=[self._obj_xy[0], self._obj_xy[1], self._table_height + 0.08],
            quat=np.asarray(self._obj_mesh2init_quat, dtype=np.float64),
        )
        body.add_joint(
            name=obj_name,
            type=mj.mjtJoint.mjJNT_FREE,
            axis=[1, 0, 0],
            pos=[0, 0, 0],
        )
        self.dict_mj_objects[obj_name] = body

        for mesh in list_meshes:
            if "collision" in mesh.name:
                body.add_geom(
                    name=mesh.name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    density=self._object_density,
                    contype=1,
                    conaffinity=1,
                    condim=6,
                    friction=[0.8, 0.005, 0.0001],
                    rgba=[0, 0, 0, 0],
                )
            else:
                body.add_geom(
                    name=mesh.name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    material=rand_mat,
                    contype=0,
                    conaffinity=0,
                    density=0,
                )

    def _place_object(self) -> None:
        for obj_name in self.dict_mj_objects:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, obj_name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            self.mj_data.qpos[qpos_adr : qpos_adr + 3] = [
                self._obj_xy[0],
                self._obj_xy[1],
                self._table_height + 0.08,
            ]
            self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = self._obj_mesh2init_quat
            self.mj_data.qvel[qpos_adr : qpos_adr + 6] = 0.0
        mj.mj_forward(self.mj_model, self.mj_data)

    def _zero_object_velocity(self) -> None:
        for obj_name in self.dict_mj_objects:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, obj_name)
            # freejoint qvel: angular (3) then linear (3)
            dof_adr = self.mj_model.jnt_dofadr[joint_id]
            self.mj_data.qvel[dof_adr : dof_adr + 6] = 0.0

    def _update_sim_env(self) -> None:
        if not self._spinning:
            return
        for obj_name in self.dict_mj_objects:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, obj_name)
            dof_adr = self.mj_model.jnt_dofadr[joint_id]
            # Keep a constant world-yaw rate; leave other components alone / zero lin.
            self.mj_data.qvel[dof_adr : dof_adr + 3] = [0.0, 0.0, self._spin_yaw_rate]
            self.mj_data.qvel[dof_adr + 3 : dof_adr + 6] = 0.0

    # ------------------------------------------------------------------
    # Sensors / LCM
    # ------------------------------------------------------------------

    def _initialize_sensors(self, sensor_config) -> None:
        if sensor_config is None:
            return
        self._dict_cam_data = {}
        self._dict_cam_info = {}
        self._dict_cam_renderer = {}

        for cam_name in self._list_camera_names:
            camera_config = sensor_config["extrinsic"][cam_name]
            self._dict_cam_data[cam_name] = RGBDData(
                camera_config["height"],
                camera_config["width"],
                depth_channel_type=DTYPE_TO_CHANNEL_TYPE[np.float32],
                name=cam_name,
            )
            self._dict_cam_info[cam_name] = CameraInfoData(
                camera_config["height"],
                camera_config["width"],
                fixed=camera_config["fixed"],
                attached_body=camera_config["attached_body"],
                depth_factor=float(camera_config.get("depth_factor", 1.0)),
                name=cam_name,
            )
            self._dict_cam_renderer[cam_name] = initialize_camera_renderer(
                self.mj_model,
                camera_config["height"],
                camera_config["width"],
            )
            self._dict_cam_info[cam_name].set_intrinsic(
                get_camera_intrinsics(
                    self.mj_model,
                    cam_name,
                    camera_config["height"],
                    camera_config["width"],
                )
            )

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
        return np.asarray(list_poses, dtype=np.float64)

    def _sync_extr_data_from_sim(self, extr_pub_que_dict: dict) -> None:
        t = self.mj_data.time

        for cam_name in self._list_camera_names:
            channel = self._dict_camera_channels.get(cam_name, cam_name)
            if channel not in extr_pub_que_dict:
                continue
            if (
                t - self._dict_last_sensor_update_time[cam_name]
                <= self._dict_sensor_update_dt[cam_name]
            ):
                continue
            self._dict_last_sensor_update_time[cam_name] = t

            cam_rgb, cam_depth = render_image_from_camera(
                self._dict_cam_renderer[cam_name],
                self.mj_data,
                cam_name,
                rgb_enable=self._dict_cam_rgb_enable[cam_name],
                depth_enable=self._dict_cam_depth_enable[cam_name],
            )
            new_data = False
            if cam_rgb is not None:
                self._dict_cam_data[cam_name].set_rgb_image(cam_rgb)
                new_data = True
            if cam_depth is not None:
                self._dict_cam_data[cam_name].set_depth_image(cam_depth)
                new_data = True
            if new_data:
                self._dict_cam_data[cam_name].set_time(t)
                extr_pub_que_dict[channel].put(
                    copy.deepcopy(self._dict_cam_data[cam_name])
                )

            self._dict_cam_info[cam_name].set_extrinsic(
                get_camera_extrinsics(self.mj_model, self.mj_data, cam_name)
            )
            self._dict_cam_info[cam_name].set_time(t)
            info_channel = channel + "_info"
            if info_channel in extr_pub_que_dict:
                extr_pub_que_dict[info_channel].put(
                    copy.deepcopy(self._dict_cam_info[cam_name])
                )

        if (
            self._obj_pose_channel in extr_pub_que_dict
            and t - self._last_obj_pose_update_time > self._obj_pose_update_dt
        ):
            self._last_obj_pose_update_time = t
            poses = self._get_object_pose_bb2world()
            if poses.size == 0:
                return
            # Publish under the same mesh name FoundationPose uses.
            name_list = [self._mesh_name]
            data = NamedVecListData(
                name=self._obj_pose_channel,
                num_vecs=1,
                vec_dim=12,
            )
            data.set_data(t=t, name_list=name_list, vec_list=poses[:1])
            extr_pub_que_dict[self._obj_pose_channel].put(copy.deepcopy(data))
