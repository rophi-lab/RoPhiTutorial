from omegaconf import DictConfig

from communication.lcm.subscriber.BaseSubManager import BaseSubManager


def get_controller_sub_manager(cfg_sub_manager: DictConfig) -> BaseSubManager:
    """Get the subscriber manager based on the configuration.
    @param[in] cfg_sub_manager: Configuration for the subscriber manager.
    @return: Subscriber manager object.
    """
    name = cfg_sub_manager["name"]

    if name == "base":
        return BaseSubManager(**cfg_sub_manager)

    elif name == "arm_hand_joint":
        from communication.lcm.subscriber.sub_manager.controller.ArmHandSubManager import (
            ArmHandSubManager,
        )

        return ArmHandSubManager(**cfg_sub_manager)

    elif name == "arm_hand_joint_col_info":
        from communication.lcm.subscriber.sub_manager.controller.ArmHandColInfoSubManager import (
            ArmHandColInfoSubManager,
        )

        return ArmHandColInfoSubManager(**cfg_sub_manager)

    elif name == "arm_hand_grasp":
        from communication.lcm.subscriber.sub_manager.controller.ArmHandGraspSubManager import (
            ArmHandGraspSubManager,
        )

        return ArmHandGraspSubManager(**cfg_sub_manager)

    elif name in (
        "joint_named_vec_list",
        "robotis_5F_hand_joint_named_vec_list",
    ):
        from communication.lcm.subscriber.sub_manager.controller.JointNamedVecListSubManager import (
            JointNamedVecListSubManager,
        )

        return JointNamedVecListSubManager(**cfg_sub_manager)

    raise ValueError(f"Unknown subscriber manager: {name}")
