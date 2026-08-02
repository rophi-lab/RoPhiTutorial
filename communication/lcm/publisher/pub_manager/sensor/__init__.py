from omegaconf import DictConfig

from communication.lcm.publisher.BasePubManager import BasePubManager
from communication.lcm.publisher.pub_manager.sensor.RealSensePubManager import (
    RealSensePubManager,
)


def get_sensor_pub_manager(cfg_pub_manager: DictConfig) -> BasePubManager:
    """
    Get the sensor publisher manager.
    @param[in] cfg_pub_manager: The configuration dictionary for the publisher manager.
    @return: The sensor publisher manager.
    """
    name = cfg_pub_manager["name"]
    if name == "base":
        return BasePubManager(**cfg_pub_manager)
    elif name == "realsense":
        return RealSensePubManager(**cfg_pub_manager)
    else:
        raise ValueError(f"Unknown sensor publisher manager: {name}")
