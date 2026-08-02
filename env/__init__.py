"""Environment factory."""

from omegaconf import DictConfig

from env.BaseEnv import BaseEnv


def get_env(cfg_env: DictConfig) -> BaseEnv:
    """Create an environment from ``cfg_env['name']``."""
    name = cfg_env["name"]

    if name == "default_mujoco":
        from env.MujocoBaseEnv import MujocoBaseEnv

        return MujocoBaseEnv(cfg_env)

    if name == "flexiv_arm_5F_hand_col":
        from env.flexiv_arm_5F_hand.FlexivArmHandColEnv import FlexivArmHandColEnv

        return FlexivArmHandColEnv(cfg_env)

    if name == "flexiv_arm_5F_hand_table_obs":
        from env.flexiv_arm_5F_hand.FlexivArmHandTableObsEnv import (
            FlexivArmHandTableObsEnv,
        )

        return FlexivArmHandTableObsEnv(cfg_env)

    if name == "flexiv_arm_5F_hand_grasp":
        from env.flexiv_arm_5F_hand.FlexivArmHandGraspEnv import FlexivArmHandGraspEnv

        return FlexivArmHandGraspEnv(cfg_env)

    if name == "mj_meshes_grasp_eval":
        from env.robotis_5f.MjMeshesGraspEvalEnv import MjMeshesGraspEvalEnv

        return MjMeshesGraspEvalEnv(cfg_env)

    if name == "table_mesh_cam":
        from env.perception_tutorial.TableMeshCamEnv import TableMeshCamEnv

        return TableMeshCamEnv(cfg_env)

    raise ValueError(f"Unknown environment name: {name}.")
