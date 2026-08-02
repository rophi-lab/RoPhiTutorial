from omegaconf import DictConfig

from communication.lcm.subscriber.BaseSubManager import BaseSubManager


def get_sensor_sub_manager(cfg_sub_manager: DictConfig) -> BaseSubManager:
    """Get the sensor subscriber manager based on the configuration.
    @return: Subscriber manager object.
    """
    name = cfg_sub_manager["name"]

    if name == "base":
        return BaseSubManager(**cfg_sub_manager)
    else:
        print(f"Unknown sensor subscriber manager: {name}, using base sub manager")
        return BaseSubManager(**cfg_sub_manager)
