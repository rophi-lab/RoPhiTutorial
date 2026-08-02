"""Flexiv arm + Robotis hand with a fixed RGB-D camera for base–cam calib."""

from __future__ import annotations

import copy

import mujoco as mj
import numpy as np
from omegaconf import DictConfig

from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.RGBDData import RGBDData
from env.MujocoBaseEnv import MujocoBaseEnv
from utils.communication.lcm.image_conversion import DTYPE_TO_CHANNEL_TYPE
from utils.mujoco.camera import (
    get_camera_extrinsics,
    get_camera_intrinsics,
    initialize_camera_renderer,
    render_image_from_camera,
)


class FlexivArmHandCamEnv(MujocoBaseEnv):
    """Publish joint_meas (via platform) plus RGBD / camera_info for calib."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        self._config = config
        self._list_camera_names = list(config["sensor"]["extrinsic"].keys())

        pub = config["pub_manager"]
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

    def initialize(self) -> None:
        print(
            "[Env] FlexivArmHandCamEnv ready. "
            "Camera publishes RGB-D for differentiable base–cam calibration."
        )

    def reset(self) -> None:
        super().reset()
        for cam_name in self._list_camera_names:
            self._dict_last_sensor_update_time[cam_name] = self.mj_data.time

    def _init_viewer(self) -> None:
        super()._init_viewer()
        with self.mj_viewer.lock():
            cam_cfg = self._config.get("viewer_cam", {}) or {}
            self.mj_viewer.cam.distance = cam_cfg.get("distance", 1.8)
            self.mj_viewer.cam.elevation = cam_cfg.get("elevation", -25)
            self.mj_viewer.cam.azimuth = cam_cfg.get("azimuth", 120)
            lookat = cam_cfg.get("lookat", [0.3, 0.0, 0.35])
            self.mj_viewer.cam.lookat = np.array(lookat)
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            self.mj_viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False

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
