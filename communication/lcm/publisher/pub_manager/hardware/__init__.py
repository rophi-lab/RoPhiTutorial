from omegaconf import DictConfig

from communication.lcm.publisher.BasePubManager import BasePubManager


def get_hardware_pub_manager(cfg_pub_manager: DictConfig) -> BasePubManager:
    """Get the publisher manager based on the configuration.
    @param[in] cfg_pub_manager: Configuration for the publisher manager.
    @return: An instance of the publisher manager.
    """
    name = cfg_pub_manager["name"]

    if name == "base":
        return BasePubManager(**cfg_pub_manager)

    elif name == "brl_arm":
        from communication.lcm.publisher.pub_manager.env.brl_arm.BrlArmPubManager import (
            BrlArmPubManager,
        )

        return BrlArmPubManager(**cfg_pub_manager)

    elif name == "brl_arm_wrist":
        from communication.lcm.publisher.pub_manager.env.brl_arm_wrist.BrlArmWristPubManager import (
            BrlArmWristPubManager,
        )

        return BrlArmWristPubManager(**cfg_pub_manager)

    elif name == "brl_arm_hand":
        from communication.lcm.publisher.pub_manager.env.brl_arm_hand.BrlArmHandPubManager import (
            BrlArmHandPubManager,
        )

        return BrlArmHandPubManager(**cfg_pub_manager)
    elif name == "brl_arm_hand_pressure":
        from communication.lcm.publisher.pub_manager.hardware.brl_arm_hand.BrlArmHandPressurePubManager import (
            BrlArmHandPressurePubManager,
        )

        return BrlArmHandPressurePubManager(**cfg_pub_manager)
    
    elif name == "brl_arm_hand":
        from communication.lcm.publisher.pub_manager.hardware.brl_arm_hand.BrlArmHandPubManager import (
            BrlArmHandPubManager,
        )

        return BrlArmHandPubManager(**cfg_pub_manager)

    elif name == "flexiv_hand":
        from communication.lcm.publisher.pub_manager.hardware.flexiv_arm_5F_hand.FlexivHandHwPubManager import (
            FlexivHandHwPubManager,
        )

        return FlexivHandHwPubManager(**cfg_pub_manager)

    raise ValueError(f"Publisher manager {name} not found.")
