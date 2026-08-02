"""
This module provides a function to create a visualizer manager based on the configuration provided.
"""

from omegaconf import DictConfig
from visualizer.BaseVisManager import BaseVisManager


def get_vis_manager(cfg_vis_manager: DictConfig) -> BaseVisManager:
    """
    Get the visualizer manager based on the configuration provided.
    @param[in] cfg_vis_manager: Configuration for the visualizer manager.
    """
    if cfg_vis_manager["name"] == "base":
        return BaseVisManager(cfg_vis_manager)
    elif cfg_vis_manager["name"].startswith("flexiv_arm_5F_hand"):
        from visualizer.vis_manager.flexiv_arm_5F_hand.FlexivArmHandVisManager import (
            FlexivArmHandVisManager,
        )

        return FlexivArmHandVisManager(cfg_vis_manager)
    elif cfg_vis_manager["name"] == "mj_eval_grasp":
        from visualizer.vis_manager.robotis_5f.MjEvalGraspVisManager import (
            MjEvalGraspVisManager,
        )

        return MjEvalGraspVisManager(cfg_vis_manager)
    elif cfg_vis_manager["name"] == "obj_pose":
        from visualizer.vis_manager.perception_tutorial.ObjPoseVisManager import (
            ObjPoseVisManager,
        )

        return ObjPoseVisManager(cfg_vis_manager)
    else:
        raise ValueError(f"Visualizer manager {cfg_vis_manager['name']} not found.")
