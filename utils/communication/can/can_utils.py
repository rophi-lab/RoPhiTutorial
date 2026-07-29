from dataclasses import dataclass
from enum import Enum

from utils.mode import HandControlMode


@dataclass(frozen=True)
class HandCANConst:
    """
    Constants for the hand CAN communication.
    These values need to match in the firmware code.
    """

    ## TODO: Current values are set for sphere sensor only. Need to update for ellipsoid sensor.

    P_MIN = -12.5
    P_MAX = 12.5
    V_MIN = -300.0
    V_MAX = 300.0
    KP_MIN = 0.0
    KP_MAX = 500.0
    KD_MIN = 0.0
    KD_MAX = 10.0
    T_MIN = -72.0
    T_MAX = 72.0

    # FT_MIN = -20.0
    # FT_MAX = 20.0
    # FN_MIN = -30.0
    # FN_MAX = 30.0
    # ANG_MIN = -135.0
    # ANG_MAX = 45.0

    FT_MIN = -10.0
    FT_MAX = 10.0
    FN_MIN = -10.0
    FN_MAX = 2.0
    THETA_ANG_MIN = -45.0
    THETA_ANG_MAX = 45.0
    PHI_ANG_MIN = -135.0
    PHI_ANG_MAX = 45.0
    CFLAG_MIN = 0.0
    CFLAG_MAX = 1.0

    SCALE = 50.0


class HandCANID(Enum):
    """
    ID for the hand CAN messages.
    """

    GRIPPER_ENABLE = 1
    MOTOR_DATA = 2
    SENSOR_DATA = 3
    LEFT_COMMAND = 4
    RIGHT_COMMAND = 5
    PRESSURE_RAW = 7


# CAN messages
HAND_MODE_MSGS = {
    HandControlMode.CURRENT_CONTROL: [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
    + ([0] * 40),
    HandControlMode.POSITION_CONTROL: [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
    + ([0] * 40),
    HandControlMode.SENSOR_DEBUG: [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFB]
    + ([0] * 40),
    HandControlMode.DISABLE_CONTROL: [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFA]
    + ([0] * 40),
}


# helper functions
def float_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    offset = x_min
    x_clip = max(min(x, x_max), x_min)
    return int(((x_clip - offset) * float((1 << bits) - 1)) / span)


def uint_to_float(x_int, x_min, x_max, bits):
    span = x_max - x_min
    offset = x_min
    x_map = float(x_int) * span / (float((1 << bits) - 1)) + offset
    return max(min(x_map, x_max), x_min)
