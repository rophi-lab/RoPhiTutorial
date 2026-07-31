"""Flexiv arm+hand grasping env: ColInfo + live object SE(3) pose + physics."""

from __future__ import annotations

import copy

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from assets.object_mesh import get_nominal_pose_to_cad
from data_type.basic_types.NamedVecListData import NamedVecListData
from data_type.basic_types.SE3PoseData import SE3PoseData
from env.flexiv_arm_5F_hand.FlexivArmHandColEnv import FlexivArmHandColEnv


class FlexivArmHandGraspEnv(FlexivArmHandColEnv):
    """Collision scene plus a freejoint ``grasp_object`` whose world SE(3) is
    published on ``object_pose_channel`` for the reactive grasp controller /
    Viser overlay. Object mass / COM / world gravity go on
    ``object_physics_channel`` for force-closure gravity compensation.

    Keys
    ----
    ``o`` : randomly re-place the object in XY (keeps sim + GUI running).
    ``r`` : full env reset (also reseeds object XY; restarts viewer).
    """

    def __init__(self, config, *args, **kwargs):
        # Must be set before ``super().__init__`` because parent ``reset()``
        # dispatches to this class.
        self._object_body_name = str(config.get("object_body_name", "grasp_object"))
        self._object_pose_channel = config["pub_manager"].get(
            "object_pose_channel", "sw_grasp_object_pose"
        )
        self._object_physics_channel = config["pub_manager"].get(
            "object_physics_channel", "sw_grasp_object_physics"
        )
        self._object_pose_dt = float(config.get("object_pose_update_dt", 1.0 / 30.0))
        self._last_object_pose_t = -1e9
        self._object_pose_data = SE3PoseData(name=self._object_pose_channel)
        self._object_physics_data = NamedVecListData(
            num_vecs=1, vec_dim=12, name=self._object_physics_channel
        )
        self._object_body_id = -1
        self._obj_name = str(config.get("obj_name", "green_bowl"))
        self._obj_spawn_xy_center = np.asarray(
            config.get("obj_spawn_xy_center", [0.48, 0.05]), dtype=np.float64
        )
        self._obj_spawn_xy_half = np.asarray(
            config.get("obj_spawn_xy_half", [0.06, 0.06]), dtype=np.float64
        )
        self._obj_spawn_z = float(config.get("obj_spawn_z", 0.12))
        self._rng = np.random.default_rng(int(config.get("obj_spawn_seed", 0)) or None)
        # CAD freejoint orientation that maps the grasp nominal/bb frame ≈ world
        # (same convention as mesh_visualizer_with_grasp_points.py).
        T_nom2cad = get_nominal_pose_to_cad(self._obj_name)
        self._R_spawn_cad = np.linalg.inv(T_nom2cad)[:3, :3].copy()

        super().__init__(config, *args, **kwargs)

        print(
            f"[GraspEnv] object={self._object_body_name} ({self._obj_name}), "
            f"pose→{self._object_pose_channel}, "
            f"physics→{self._object_physics_channel}. "
            "Key 'o' = random XY respawn (no reset); 'r' = full reset."
        )

    def reset(self) -> None:
        super().reset()
        self._object_body_id = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_BODY, self._object_body_name
        )
        if self._object_body_id < 0:
            raise ValueError(
                f"Body {self._object_body_name!r} missing in grasping scene XML"
            )
        self._reseed_object_pose()

    def _handle_key(self, key: str) -> None:
        if key in ("o", "O"):
            self._reseed_object_pose()
            return
        super()._handle_key(key)

    def _key_callback(self, keycode) -> None:
        try:
            ch = chr(keycode)
        except (ValueError, OverflowError):
            ch = ""
        if ch in ("o", "O"):
            self._reseed_object_pose()
            return
        super()._key_callback(keycode)

    def _reseed_object_pose(self) -> None:
        """Random XY (+ small yaw) respawn; keep robot / viewer running."""
        if self._object_body_id < 0:
            self._object_body_id = mj.mj_name2id(
                self.mj_model, mj.mjtObj.mjOBJ_BODY, self._object_body_name
            )
        if self._object_body_id < 0:
            print(f"[GraspEnv] body {self._object_body_name!r} not found")
            return

        jid = self.mj_model.body_jntadr[self._object_body_id]
        if jid < 0:
            return
        qadr = int(self.mj_model.jnt_qposadr[jid])
        dadr = int(self.mj_model.jnt_dofadr[jid])

        xy = self._obj_spawn_xy_center + self._rng.uniform(
            -self._obj_spawn_xy_half, self._obj_spawn_xy_half
        )
        yaw = float(self._rng.uniform(-0.4, 0.4))
        R = Rotation.from_euler("z", yaw).as_matrix() @ self._R_spawn_cad
        quat = Rotation.from_matrix(R).as_quat()  # xyzw

        def _apply():
            # MuJoCo freejoint: qpos [x y z qw qx qy qz], qvel 6-dof.
            self.mj_data.qpos[qadr : qadr + 3] = [xy[0], xy[1], self._obj_spawn_z]
            self.mj_data.qpos[qadr + 3 : qadr + 7] = [
                quat[3],
                quat[0],
                quat[1],
                quat[2],
            ]
            self.mj_data.qvel[dadr : dadr + 6] = 0.0
            mj.mj_forward(self.mj_model, self.mj_data)

        viewer = getattr(self, "mj_viewer", None)
        if viewer is not None and viewer.is_running():
            with viewer.lock():
                _apply()
        else:
            _apply()

        # Force next sync to publish the new pose immediately.
        self._last_object_pose_t = -1e9
        print(
            f"[GraspEnv] object respawned at xy=({xy[0]:.3f}, {xy[1]:.3f}), "
            f"yaw={yaw:.2f} rad"
        )

    def _get_object_T(self) -> np.ndarray:
        """World pose of ``grasp_object`` as a 4x4 matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self.mj_data.xpos[self._object_body_id].copy()
        T[:3, :3] = self.mj_data.xmat[self._object_body_id].reshape(3, 3).copy()
        return T

    def _get_object_physics_vec(self) -> np.ndarray:
        """``[mass, com_cad(3), gravity_world(3), pad(5)]`` matching tutorial 11."""
        physics = np.zeros(12, dtype=np.float64)
        bid = self._object_body_id
        if bid < 0:
            return physics
        physics[0] = float(self.mj_model.body_mass[bid])
        physics[1:4] = np.asarray(self.mj_model.body_ipos[bid], dtype=np.float64)
        physics[4:7] = np.asarray(self.mj_model.opt.gravity, dtype=np.float64)
        return physics

    def _sync_data_from_sim(self, intr_pub_que_dict, extr_pub_que_dict, time_pub_que):
        super()._sync_data_from_sim(intr_pub_que_dict, extr_pub_que_dict, time_pub_que)
        t = float(self.mj_data.time)
        if t - self._last_object_pose_t < self._object_pose_dt:
            return
        self._last_object_pose_t = t

        if self._object_pose_channel in extr_pub_que_dict:
            self._object_pose_data.set_from_matrix(t, self._get_object_T())
            try:
                while not extr_pub_que_dict[self._object_pose_channel].empty():
                    extr_pub_que_dict[self._object_pose_channel].get_nowait()
            except Exception:
                pass
            extr_pub_que_dict[self._object_pose_channel].put(
                copy.deepcopy(self._object_pose_data)
            )

        if self._object_physics_channel in extr_pub_que_dict:
            physics = self._get_object_physics_vec()
            self._object_physics_data.set_data(
                t, ["physics_params"], physics.reshape(1, -1)
            )
            try:
                while not extr_pub_que_dict[self._object_physics_channel].empty():
                    extr_pub_que_dict[self._object_physics_channel].get_nowait()
            except Exception:
                pass
            extr_pub_que_dict[self._object_physics_channel].put(
                copy.deepcopy(self._object_physics_data)
            )
