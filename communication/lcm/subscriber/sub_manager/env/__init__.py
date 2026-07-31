from omegaconf import DictConfig

from communication.lcm.subscriber.BaseSubManager import BaseSubManager


def get_env_sub_manager(cfg_sub_manager: DictConfig) -> BaseSubManager:
    """Get the subscriber manager based on the configuration.
    @param[in] cfg_sub_manager: Configuration for the subscriber manager.
    @return: Subscriber manager object.
    """
    name = cfg_sub_manager["name"]

    if name == "base":
        return BaseSubManager(**cfg_sub_manager)
    elif name == "joint":
        from communication.lcm.subscriber.sub_manager.env.JointSubManager import (
            JointSubManager,
        )

        return JointSubManager(**cfg_sub_manager)

    elif name == "joint_named_vec_list":
        from communication.lcm.subscriber.sub_manager.env.JointNamedVecListSubManager import (
            JointNamedVecListSubManager,
        )

        return JointNamedVecListSubManager(**cfg_sub_manager)

    raise ValueError(f"Unknown subscriber manager: {name}")
