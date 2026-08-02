from omegaconf import DictConfig

from communication.lcm.publisher.BasePubManager import BasePubManager
from communication.lcm.publisher.pub_manager.perception.Point3DPubManager import (
    Point3DPubManager,
)
from communication.lcm.publisher.pub_manager.perception.NamedVecPubManager import (
    NamedVecPubManager,
)
from communication.lcm.publisher.pub_manager.perception.MaskPubManager import (
    MaskPubManager,
)


def get_percep_pub_manager(cfg_sub_manager: DictConfig) -> BasePubManager:
    """Get the perception publisher manager based on the configuration.
    @return: Publisher manager object.
    """
    name = cfg_sub_manager["name"]

    if name == "base":
        return BasePubManager(**cfg_sub_manager)
    elif name == "point3d":
        return Point3DPubManager(**cfg_sub_manager)
    elif name == "named_vec":
        return NamedVecPubManager(**cfg_sub_manager)
    elif name == "mask":
        return MaskPubManager(**cfg_sub_manager)
    raise ValueError(f"Unknown perception publisher manager: {name}")
