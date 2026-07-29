from omegaconf import DictConfig
from omegaconf import OmegaConf

from robot_platform.hardware import BaseHardwarePlatform


def get_hardware_platform(
    cfg_platform: DictConfig,
) -> BaseHardwarePlatform:
    """
    Set the platform based on the configuration.
    @param[in] cfg_platform: Configuration for the platform.
    """
    name = cfg_platform["name"]
    # merged to get channel names from sub and pub manager configs
    # cfg_merged = OmegaConf.merge(cfg_platform, cfg_sub_manager, cfg_pub_manager)
    if name == "base":
        return BaseHardwarePlatform(cfg_platform)
    elif name == "brl_arm_wrist":
        from robot_platform.hardware.brl_arm_wrist.HardwareBrlArmWrist import (
            HardwareBrlArmWrist,
        )

        return HardwareBrlArmWrist(cfg_platform)
    elif name == "brl_arm_hand":
        from robot_platform.hardware.brl_arm_hand.HardwareBrlArmHand import (
            HardwareBrlArmHand,
        )

        return HardwareBrlArmHand(cfg_platform)

    elif name == "brl_hand_only":
        from robot_platform.hardware.brl_arm_hand.HardwareBrlHandOnly import (
            HardwareBrlHandOnly,
        )

        return HardwareBrlHandOnly(cfg_platform)
    elif name == "brl_arm_3f_hand":
        from robot_platform.hardware.brl_arm_3F_hand.HardwareBrlArm3FHand import (
            HardwareBrlArm3FHand,
        )

        return HardwareBrlArm3FHand(cfg_platform)
    elif name == "brl_3f_hand_only":
        from robot_platform.hardware.brl_arm_3F_hand.HardwareBrl3FHandOnly import (
            HardwareBrl3FHandOnly,
        )

        return HardwareBrl3FHandOnly(cfg_platform)
    
    elif name == "brl_5f_hand_only":
        from robot_platform.hardware.brl_arm_5F_hand.HardwareBrl5FHandOnly import (
            HardwareBrl5FHandOnly,
        )

        return HardwareBrl5FHandOnly(cfg_platform)
    
    elif name == "brl_arm_5f_hand":
        from robot_platform.hardware.brl_arm_5F_hand.HardwareBrlArm5FHand import (
            HardwareBrlArm5FHand,
        )

        return HardwareBrlArm5FHand(cfg_platform)
    elif name == "flexiv_arm":
        from robot_platform.hardware.flexiv_arm.HardwareFlexivArm import (
            HardwareFlexivArm,
        )

        return HardwareFlexivArm(cfg_platform)
    elif name == "flexiv_arm_5f_hand":
        from robot_platform.hardware.flexiv_arm_5F_hand.HardwareFlexivArm5FHand import (
            HardwareFlexivArm5FHand,
        )

        return HardwareFlexivArm5FHand(cfg_platform)
    else:
        raise ValueError(f"Unknown platform name: {name}")
