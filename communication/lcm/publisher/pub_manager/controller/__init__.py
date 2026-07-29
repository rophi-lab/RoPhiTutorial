from omegaconf import DictConfig

from communication.lcm.publisher.BasePubManager import BaseCtrlPubManager


def get_controller_pub_manager(cfg_pub_manager: DictConfig) -> BaseCtrlPubManager:
    """Get the publisher manager based on the configuration.
    @param[in] cfg_pub_manager: Configuration for the publisher manager.
    @return: Publisher manager object.
    """
    name = cfg_pub_manager["name"]

    if name == "base":
        return BaseCtrlPubManager(**cfg_pub_manager)

    elif name == "joint_ctrl":
        from communication.lcm.publisher.pub_manager.controller.JointCtrlPubManager import (
            JointCtrlPubManager,
        )

        return JointCtrlPubManager(**cfg_pub_manager)

    elif name == "joint_ctrl_pose":
        from communication.lcm.publisher.pub_manager.controller.JointCtrlPosePubManager import (
            JointCtrlPosePubManager,
        )

        return JointCtrlPosePubManager(**cfg_pub_manager)

    elif name == "joint_ctrl_traj":
        from communication.lcm.publisher.pub_manager.controller.JointCtrlTrajPubManager import (
            JointCtrlTrajPubManager,
        )

        return JointCtrlTrajPubManager(**cfg_pub_manager)

    elif name == "joint_ctrl_imp":
        from communication.lcm.publisher.pub_manager.controller.JointCtrlImpPubManager import (
            JointCtrlImpPubManager,
        )

        return JointCtrlImpPubManager(**cfg_pub_manager)

    elif name == "joint_ctrl_grasp_viz":
        from communication.lcm.publisher.pub_manager.controller.JointCtrlGraspVizPubManager import (
            JointCtrlGraspVizPubManager,
        )

        return JointCtrlGraspVizPubManager(**cfg_pub_manager)

    raise ValueError(f"Unknown publisher manager: {name}")
