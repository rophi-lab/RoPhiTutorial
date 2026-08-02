from sensor.BaseSensorManager import BaseSensorManager
from sensor.sensor.realsense.RealSense import RealSense
from sensor.sensor_info.realsense.CameraInfo import CameraInfo


class RealSenseManager(BaseSensorManager):
    """
    RealSenseManager is a class that manages the RealSense sensors.
    It inherits from the BaseSensorManager class.
    """

    def __init__(self, config, *args, **kwargs):
        """
        Initialize the RealSenseManager.
        @param[in] config: The configuration dictionary.
        """
        super().__init__(config, *args, **kwargs)
        self._config = config
        self._realsense_configs = self._config["realsense"]
        self._num_realsense = len(self._realsense_configs)
        self._pub_manager_config = self._config["pub_manager"]

    def initialize_sensors_and_sensor_info(self):
        """
        Initialize the sensors and sensor info.
        """
        # Create the sensors
        for key, cfg in self._realsense_configs.items():
            name = self._pub_manager_config["realsense"][f"{key}_channel"]
            serial = str(cfg["serial"])
            fixed = cfg.get("fixed", False)
            attached_body = cfg.get("attached_body", None)
            width = cfg.get("width", 640)
            height = cfg.get("height", 480)
            frequency = cfg.get("frequency", 30)
            fx = cfg.get("fx", 0)
            fy = cfg.get("fy", 0)
            cx = cfg.get("cx", 0)
            cy = cfg.get("cy", 0)
            depth_factor = cfg.get("depth_factor", 1000)
            print(depth_factor)
            tf_link_to_cam = cfg.get("tf_link_to_cam", None)
            visualize = cfg.get("visualize", False)
            visual_preset = cfg.get("visual_preset", None)
            log_latency = cfg.get("log_latency", False)
            self._sensors[name] = RealSense(
                name,
                serial,
                height=height,
                width=width,
                frequency=frequency,
                visualize=visualize,
                visual_preset=visual_preset,
                log_latency=log_latency,
            )
            self._sensor_info[name] = CameraInfo(
                name + "_info",
                serial,
                fixed,
                attached_body,
                height,
                width,
                frequency,
                fx,
                fy,
                cx,
                cy,
                depth_factor,
                tf_link_to_cam=tf_link_to_cam,
                channel_name=name + "_info",
            )
            # print(self._sensor_info[name].get_depth_factor())
