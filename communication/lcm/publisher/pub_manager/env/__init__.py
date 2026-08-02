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

    elif name == "joint_named_vec_list":
        from communication.lcm.publisher.pub_manager.env.JointNamedVecListPubManager import (
            JointNamedVecListPubManager,
        )

        return JointNamedVecListPubManager(**cfg_pub_manager)

    elif name == "joint_two_named_vec_lists":
        from communication.lcm.publisher.pub_manager.env.JointTwoNamedVecListsPubManager import (
            JointTwoNamedVecListsPubManager,
        )

        return JointTwoNamedVecListsPubManager(**cfg_pub_manager)

    elif name == "cam_only":
        from communication.lcm.publisher.pub_manager.env.CamOnlyPubManager import (
            CamOnlyPubManager,
        )

        return CamOnlyPubManager(**cfg_pub_manager)

    elif name == "flexiv_arm_hand_cam":
        from communication.lcm.publisher.pub_manager.env.flexiv_arm_5F_hand.FlexivArmHandCamPubManager import (
            FlexivArmHandCamPubManager,
        )

        return FlexivArmHandCamPubManager(**cfg_pub_manager)

    raise ValueError(f"Publisher manager {name} not found.")
