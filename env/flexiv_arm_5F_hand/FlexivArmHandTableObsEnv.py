"""Flexiv arm + 5F hand with a table and obstacles (static ColInfo from MJCF)."""

from env.flexiv_arm_5F_hand.FlexivArmHandColEnv import FlexivArmHandColEnv


class FlexivArmHandTableObsEnv(FlexivArmHandColEnv):
    """Collision env whose scene places a table + named ``obs_*`` boxes in front
    of the arm (see ``assets/scene/flexiv_arm/hand_table_obstacles.xml``).

    Static ColInfo is still derived automatically from MJCF geoms named
    ``floor`` / ``wall_*`` / ``table`` / ``table_*`` / ``obs_*`` via
    :func:`utils.collision.wall_geoms.extract_static_col_geoms`.
    """
