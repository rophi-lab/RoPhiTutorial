from enum import Enum


class EnvMode(Enum):
    """
    Enum for the different modes of the environment."""

    SIM_REAL_TIME = 0
    SIM_SIM_TIME = 1
    REAL = 2


class HardwareMode(Enum):
    """
    Enum for the arm mode.
    """

    ESTOP = 0
    HOME = 1
    GO_HOME = 2
    HOLD = 3
    READY = 4
    CONTROL = 5
    APPLY_LAST_CONTROL = 6


class HandControlMode(Enum):
    """
    Enum for the hand control mode.
    """

    CURRENT_CONTROL = 0
    POSITION_CONTROL = 1
    SENSOR_DEBUG = 2
    # disable control is only used to turn off the hand
    DISABLE_CONTROL = 3


class VisMode(Enum):
    """
    Enum for the visualization."""

    OFF = 0
    ON = 1
