"""Controller factory."""

from omegaconf import DictConfig

from controller.BaseController import BaseController


def get_controller(cfg_controller: DictConfig) -> BaseController:
    """Create a controller from ``cfg_controller['name']``."""
    name = cfg_controller["name"]

    if name == "base":
        return BaseController(cfg_controller)

    if name == "grav_comp":
        from controller.GravCompControl import GravCompController

        return GravCompController(cfg_controller)

    if name == "col_aware_grav_comp":
        from controller.CollisionAwareGravCompControl import (
            CollisionAwareGravCompController,
        )

        return CollisionAwareGravCompController(cfg_controller)

    if name == "ik_qp":
        from controller.IKQPControl import IKQPController

        return IKQPController(cfg_controller)

    if name == "min_jerk_p2p":
        from controller.MinimumJerkP2PControl import MinimumJerkP2PController

        return MinimumJerkP2PController(cfg_controller)

    if name == "impedance":
        from controller.ImpedanceControl import ImpedanceController

        return ImpedanceController(cfg_controller)

    if name == "ik_reach":
        from controller.IKReachControl import IKReachController

        return IKReachController(cfg_controller)

    if name == "three_finger_reactive_grasp":
        from controller.ThreeFingerReactiveGrasping import ThreeFingerReactiveGrasping

        return ThreeFingerReactiveGrasping(cfg_controller)

    if name in (
        "three_finger_reactive_force_closure",
        "ThreeFingerReactiveForceClosureGrasping",
    ):
        from controller.ThreeFingerReactiveForceClosureGrasping import (
            ThreeFingerReactiveForceClosureGrasping,
        )

        return ThreeFingerReactiveForceClosureGrasping(cfg_controller)

    if name in ("mj_eval_grasp", "mj_eval_grasp_control", "MjEvalGraspControl"):
        from controller.MjEvalGraspControl import MjEvalGraspControl

        return MjEvalGraspControl(cfg_controller)

    raise ValueError(f"Unknown controller name: {name}.")
