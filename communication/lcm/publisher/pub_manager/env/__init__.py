from omegaconf import DictConfig

from communication.lcm.publisher.BasePubManager import BasePubManager


def get_env_pub_manager(cfg_pub_manager: DictConfig) -> BasePubManager:
    """Get the publisher manager based on the configuration.
    @param[in] cfg_pub_manager: Configuration for the publisher manager.
    @return: An instance of the publisher manager.
    """
    name = cfg_pub_manager["name"]

    if name == "base":
        return BasePubManager(**cfg_pub_manager)

    elif name == "flexiv_arm_hand":
        from communication.lcm.publisher.pub_manager.env.flexiv_arm_5F_hand.FlexivArmHandPubManager import (
            FlexivArmHandPubManager,
        )

        return FlexivArmHandPubManager(**cfg_pub_manager)
    elif name == "flexiv_arm_hand_col_info":
        from communication.lcm.publisher.pub_manager.env.flexiv_arm_5F_hand.FlexivArmHandColInfoPubManager import (
            FlexivArmHandColInfoPubManager,
        )

        return FlexivArmHandColInfoPubManager(**cfg_pub_manager)

    elif name == "flexiv_arm_hand_grasp":
        from communication.lcm.publisher.pub_manager.env.flexiv_arm_5F_hand.FlexivArmHandGraspPubManager import (
            FlexivArmHandGraspPubManager,
        )

        return FlexivArmHandGraspPubManager(**cfg_pub_manager)

    raise ValueError(f"Publisher manager {name} not found.")
