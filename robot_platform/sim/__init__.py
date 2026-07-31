"""Factory for MuJoCo simulation platforms."""

from omegaconf import DictConfig, OmegaConf

from robot_platform.sim.BaseSimPlatform import BaseSimPlatform


def get_sim_platform(
    cfg_platform: DictConfig,
    cfg_sub_manager: DictConfig = OmegaConf.create(),
    cfg_pub_manager: DictConfig = OmegaConf.create(),
) -> BaseSimPlatform:
    """Build a sim platform from config.

    Channel names from sub/pub manager configs are merged into the platform
    config so constructors can pick them up as kwargs.
    """
    name = cfg_platform["name"]
    cfg = OmegaConf.merge(cfg_platform, cfg_sub_manager, cfg_pub_manager)

    if name == "flexiv_arm_5f_hand":
        from robot_platform.sim.flexiv_arm_5F_hand.SimFlexivArm5FHand import (
            SimFlexivArm5FHand,
        )

        return SimFlexivArm5FHand(**cfg)

    if name == "robotis_5f_hand":
        from robot_platform.sim.robotis_5F_hand.SimRobotis5FHand import SimRobotis5FHand

        return SimRobotis5FHand(**cfg)

    raise ValueError(f"Unknown platform name: {name}")
