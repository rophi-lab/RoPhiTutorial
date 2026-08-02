from omegaconf import DictConfig

from communication.lcm.subscriber.BaseSubManager import BaseSubManager
from communication.lcm.subscriber.sub_manager.perception.SingleRGBDSubManager import (
    SingleRGBDSubManager,
)
from communication.lcm.subscriber.sub_manager.perception.MultiRGBDSubManager import (
    MultiRGBDSubManager,
)
from communication.lcm.subscriber.sub_manager.perception.MultiRGBDMaskSubManager import (
    MultiRGBDMaskSubManager,
)


def get_percep_sub_manager(cfg_sub_manager: DictConfig) -> BaseSubManager:
    """Get the perception subscriber manager based on the configuration.
    @return: Subscriber manager object.
    """
    name = cfg_sub_manager["name"]

    if name == "base":
        return BaseSubManager(**cfg_sub_manager)
    elif name == "single_rgbd":
        return SingleRGBDSubManager(**cfg_sub_manager)
    elif name == "multi_rgbd":
        return MultiRGBDSubManager(**cfg_sub_manager)
    elif name == "multi_rgbd_mask":
        return MultiRGBDMaskSubManager(**cfg_sub_manager)
    raise ValueError(f"Unknown perception subscriber manager: {name}")
