from omegaconf import DictConfig

from sensor.BaseSensorManager import BaseSensorManager


def get_sensor_manager(cfg_sensors: DictConfig) -> BaseSensorManager:
    """
    Get the sensor manager based on the configuration.
    @param[in] cfg_sensors: Configuration for the sensors.
    @return: The sensor manager.
    """
    name = cfg_sensors["name"]

    if name == "base":
        return BaseSensorManager(cfg_sensors)
    elif name == "realsense":
        from sensor.sensor_manager.RealSenseManager import RealSenseManager

        return RealSenseManager(cfg_sensors)
    raise ValueError(f"Unknown sensor manager name: {name}")
