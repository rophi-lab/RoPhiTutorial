"""MuJoCo-backed environment with optional passive viewer."""

from __future__ import annotations

import time
from queue import Queue

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from omegaconf import DictConfig

from env.BaseEnv import BaseEnv
from utils.mode import VisMode
from utils.mujoco.collision import check_collision


class MujocoBaseEnv(BaseEnv):
    """Load an MJCF scene, step MuJoCo, and optionally show a viewer."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        self.mj_viewer = None
        self.mj_model = None
        self.mj_data = None
        self.mj_spec = None

        super().__init__(config, *args, **kwargs)

        self._scene_xml_path = config["scene_xml_path"]
        self._scene_params_config = config.get("scene_params", None)
        self._sensor_config = config.get("sensor", None)

        self._load_model_and_gen_scene(self._scene_params_config, self._scene_xml_path)
        self.reset()

        if self.vis_mode == VisMode.ON:
            self._init_viewer()

        self.mj_model.opt.timestep = self._sim_dt
        self._qd_prev = self.mj_data.qvel.copy()

    def __del__(self):
        viewer = getattr(self, "mj_viewer", None)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    def stop(self) -> None:
        """Close the MuJoCo viewer if it is open."""
        if self.mj_viewer is not None:
            self.mj_viewer.close()
            self.mj_viewer = None
            print("[Env] Viewer closed.")

    def reset(self) -> None:
        """Reset sim state, platform, and (if needed) the viewer."""
        self._pause = False
        self._finished = False
        self._last_real_t = time.time()
        self._last_view_t = time.time()

        self.platform.reset(self.mj_data)
        mj.mj_forward(self.mj_model, self.mj_data)
        check_collision(self.mj_model, self.mj_data)
        self._initialize_sensors(self._sensor_config)

        if self.vis_mode == VisMode.ON and self.mj_viewer is not None:
            self.mj_viewer.close()
            self._init_viewer()

        print("[Env] Environment reset complete.")

    def _load_model_and_gen_scene(self, scene_params_config, scene_xml_path) -> None:
        """Load MJCF, allow subclass edits, then compile the model."""
        self.mj_spec = mj.MjSpec.from_file(scene_xml_path)
        self._modify_mjspec()
        self.mj_model = self.mj_spec.compile()
        self.mj_data = mj.MjData(self.mj_model)
        self.platform.set_mj_data_name_idx(self.mj_data)

    def _initialize_sensors(self, sensor_config) -> None:
        """Initialize sensors (cameras, etc.). Override in subclasses."""

    def _modify_mjspec(self) -> None:
        """Optionally edit ``self.mj_spec`` before compile. Override as needed."""

    def _init_viewer(self) -> None:
        self.mj_viewer = mjv.launch_passive(
            self.mj_model,
            self.mj_data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self._key_callback,
        )
        with self.mj_viewer.lock():
            self.mj_viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTSPLIT] = True
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_SELECT] = True
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = False
            self.mj_viewer.opt.geomgroup[3] = True

            self.mj_viewer.cam.distance = 2
            self.mj_viewer.cam.elevation = -90
            self.mj_viewer.cam.azimuth = 0
            self.mj_viewer.cam.lookat = np.array([0.0, 0.0, 0.0])

            mj.mj_forward(self.mj_model, self.mj_data)

        print("[Env] Viewer started.")

    def _check_stopping_condition(self) -> None:
        if self.vis_mode == VisMode.ON and not self.mj_viewer.is_running():
            self._finished = True
            print("[Env] Viewer closed. Exiting.")

    def _step_simulation(self) -> None:
        for _ in range(self._sim_steps_per_control):
            qdd = (self.mj_data.qvel - self._qd_prev) / self._sim_dt
            self.mj_data.qfrc_applied = 0.0
            self.platform.add_friction_and_inertial_correction_to_sim(self.mj_data, qdd)
            self._qd_prev = self.mj_data.qvel.copy()
            mj.mj_step(self.mj_model, self.mj_data)

    def _apply_sim_control(self, ctrl_data) -> None:
        self.platform.apply_sim_control(ctrl_data, self.mj_data)

    def _publish_sim_time(self, time_pub_que: Queue) -> None:
        time_pub_que.put(self.mj_data.time)

    def _sync_data_from_hardware(
        self,
        ctrl_sub_que: Queue,
        intr_sub_que_dict: dict,
        extr_sub_que_dict: dict,
        intr_pub_que_dict: dict,
        extr_pub_que_dict: dict,
        time_pub_que,
    ) -> None:
        """REAL mode hook. Default: no-op."""

    def _sync_data_from_sim(
        self,
        intr_pub_que_dict: dict,
        extr_pub_que_dict: dict,
        time_pub_que: Queue,
    ) -> None:
        self.platform.sync_intr_data_from_sim(self.mj_data, intr_pub_que_dict)
        self._sync_extr_data_from_sim(extr_pub_que_dict)

    def _sync_extr_data_from_sim(self, extr_pub_que_dict: dict) -> None:
        """Publish extrinsic sim data (objects, cameras, etc.). Default: no-op."""

    def _update_vis(self) -> None:
        if self.mj_viewer is not None and self.mj_viewer.is_running():
            self.mj_viewer.sync()

    def _key_callback(self, keycode) -> None:
        if chr(keycode) == " ":
            self._pause = not self._pause
            print(f"[Env] Paused: {self._pause}")
