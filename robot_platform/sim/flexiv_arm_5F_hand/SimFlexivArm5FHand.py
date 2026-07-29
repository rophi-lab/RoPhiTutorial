import copy

import numpy as np

import mujoco as mj

from robot_platform.sim.BaseSimPlatform import BaseSimPlatform
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData


_NUM_ARM_JOINTS = 7
_NUM_HAND_JOINTS = 20
_NUM_JOINTS = _NUM_ARM_JOINTS + _NUM_HAND_JOINTS

# Collision URDF (same combined model the controller/gravity comp use). Frame
# names here (link1..link7, palm) must match the pinocchio BODY frames the
# collision-aware controller does FK on.
_COL_URDF_PATH = "assets/scene/flexiv_arm/urdf/Rizon4_viser.urdf"
# Cylinder radii (m) approximating each arm link, proximal -> distal.
_ARM_LINK_RADII = [0.075, 0.07, 0.07, 0.07, 0.07, 0.065, 0.05]
# The last link's kinematic segment (link7 -> palm) is only ~0.10 m, which
# leaves the wrist/flange/hand-base region under-covered. Extend that cylinder
# past the palm so the end assembly is properly enclosed for collision checks.
_LAST_LINK_EXTENSION = 0.12


def _align_z_to(u: np.ndarray) -> np.ndarray:
    """Rotation whose +z axis points along unit-ish vector u (for FCL cylinders,
    which are defined along local z)."""
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    c = float(z @ u)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _build_hand_collision_primitives():
    """Per-finger-link collision primitives for the Robotis RH-5 hand (palm box
    + one cylinder per distal 2 links of each finger), identical to
    SimBrlArm5FHand.get_col_info_data's hand section -- same hand, same URDF
    link names/geometry, so the offsets carry over unchanged."""
    return [
        {
            "name": "palm",
            "geom_type": "box",
            "size": np.array([0.045, 0.09, 0.15]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.05],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        # Thumb
        {
            "name": "finger_r_link_1_thumb3",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.034]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_1_thumb4",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.036]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        # Index
        {
            "name": "finger_r_link_2_index2",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.03]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_2_index3",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.034]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_2_index4",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.036]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        # Middle
        {
            "name": "finger_r_link_3_middle2",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.03]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_3_middle3",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.034]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_3_middle4",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.036]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        # Ring
        {
            "name": "finger_r_link_4_ring2",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.03]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_4_ring3",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.034]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_4_ring4",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.036]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        # Little
        {
            "name": "finger_r_link_5_little2",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.03]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_5_little3",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.034]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "finger_r_link_5_little4",
            "geom_type": "cylinder",
            "size": np.array([0.013, 0.036]),
            "offset": np.array(
                [
                    [1.0, 0.0, 0.0, -0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.015],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    ]


def _build_arm_collision_primitives():
    """Per-link collision primitives for the arm + hand, computed from the URDF
    kinematics. Each arm link is a cylinder spanning to its child joint (in the
    link-local frame); the hand is the same per-finger-link primitive set as
    SimBrlArm5FHand (palm box + one cylinder per distal 2 links per finger).
    Returns the list of dicts consumed by ColInfoData.add_col_geom.

    Offsets are LINK-LOCAL SE3 (the controller left-multiplies by the link's
    world pose from FK), matching SimBrlArmHand.get_col_info_data.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(_COL_URDF_PATH)
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    fid = {f.name: i for i, f in enumerate(model.frames)}

    def oMf(name):
        return data.oMf[fid[name]]

    links = [f"link{i}" for i in range(1, _NUM_ARM_JOINTS + 1)]
    chain = links + ["palm"]
    prims = []
    for i, link in enumerate(links):
        # Child joint origin expressed in this link's frame (fixed offset).
        v = (oMf(link).inverse() * oMf(chain[i + 1])).translation
        seg_len = float(np.linalg.norm(v))
        u = v / max(seg_len, 1e-9)  # unit direction toward the child (distal)
        if i == len(links) - 1:
            # Extend the PROXIMAL end (back toward the wrist), keeping the distal
            # end at the palm, so the cylinder covers the wrist/flange region
            # instead of poking past the hand.
            length = seg_len + _LAST_LINK_EXTENSION
            center = seg_len / 2.0 - _LAST_LINK_EXTENSION / 2.0
        else:
            length = seg_len
            center = seg_len / 2.0
        offset = np.eye(4)
        # offset[:3, :3] = _align_z_to(u)
        offset[:3, 3] = u * center  # cylinder center along the link axis
        prims.append(
            {
                "name": link,
                "geom_type": "cylinder",
                "size": np.array([_ARM_LINK_RADII[i], length]),
                "offset": offset,
            }
        )
    # Hand: per-finger-link primitives (palm box + 2 cylinders per finger),
    # identical to SimBrlArm5FHand -- same hand, same URDF link names.
    prims.extend(_build_hand_collision_primitives())
    return prims


_ARM_JOINT_NAMES = [f"joint{i + 1}" for i in range(_NUM_ARM_JOINTS)]

# Robotis RH-5 five-finger hand joints, in the same order the URDF declares them
# (5 fingers x 4 joints). Names carry the a08..a27 actuator prefixes because
# that is exactly how they are named in Rizon4_hand_robot.xml.
_HAND_JOINT_NAMES = [
    "a08_finger_r_joint_1_thumb1",
    "a09_finger_r_joint_1_thumb2",
    "a10_finger_r_joint_1_thumb3",
    "a11_finger_r_joint_1_thumb4",
    "a12_finger_r_joint_2_index1",
    "a13_finger_r_joint_2_index2",
    "a14_finger_r_joint_2_index3",
    "a15_finger_r_joint_2_index4",
    "a16_finger_r_joint_3_middle1",
    "a17_finger_r_joint_3_middle2",
    "a18_finger_r_joint_3_middle3",
    "a19_finger_r_joint_3_middle4",
    "a20_finger_r_joint_4_ring1",
    "a21_finger_r_joint_4_ring2",
    "a22_finger_r_joint_4_ring3",
    "a23_finger_r_joint_4_ring4",
    "a24_finger_r_joint_5_little1",
    "a25_finger_r_joint_5_little2",
    "a26_finger_r_joint_5_little3",
    "a27_finger_r_joint_5_little4",
]


class SimFlexivArm5FHand(BaseSimPlatform):
    """MuJoCo platform for the 7-DoF Flexiv Rizon4 arm + 20-DoF Robotis RH-5 hand.

    Mirrors SimBrlArm5FHand's monolithic structure (one 27-DoF command in,
    two measurement channels out) but drives the arm exactly like SimFlexivArm
    (torque + gravity comp) and the hand like SimRobotis5FHand (pure joint-space
    PD), rather than going through the BRL arm's motor-space conversion.

    Command / measurement layout is [arm(7), hand(20)] = 27, matching the
    Rizon4_hand_robot.xml joint order.
    """

    def __init__(
        self,
        *args,
        arm_joint_meas_freq: int = 1000,
        hand_joint_meas_freq: int = 1000,
        default_q_values=None,
        arm_joint_meas_channel: str = "hw_flexiv_arm_joint_meas",
        hand_joint_meas_channel: str = "hw_robotis_5F_hand_joint_meas",
        gravity_comp: bool = True,
        default_hand_hold_kp: float = 1.0,
        default_hand_hold_kd: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if default_q_values is None:
            default_q_values = [0.0] * _NUM_JOINTS
        if len(default_q_values) != _NUM_JOINTS:
            raise ValueError(
                f"default_q_values must have {_NUM_JOINTS} entries "
                f"([arm(7), hand(20)]), got {len(default_q_values)}."
            )

        self.arm_joint_name = list(_ARM_JOINT_NAMES)
        self.hand_joint_name = list(_HAND_JOINT_NAMES)

        self.arm_joint_meas_data = JointMeasData(num_joints=_NUM_ARM_JOINTS)
        self.hand_joint_meas_data = JointMeasData(num_joints=_NUM_HAND_JOINTS)

        self._default_q = np.array(default_q_values, dtype=np.float64)
        self._last_ctrl_data = JointCtrlData(num_joints=_NUM_JOINTS)

        self._gravity_comp = gravity_comp
        self._default_hand_hold_kp = default_hand_hold_kp
        self._default_hand_hold_kd = default_hand_hold_kd

        # Gates the startup-only arm gravity-comp fallback (see apply_sim_control).
        # Flipped True the first time a real controller command arrives.
        self._ever_received_ctrl = False

        self._arm_joint_meas_channel = arm_joint_meas_channel
        self._hand_joint_meas_channel = hand_joint_meas_channel

        self._list_of_arm_joint_indices = []
        self._list_of_hand_joint_indices = []
        self._list_of_joint_indices = []

        self._last_pub_time = {
            self._arm_joint_meas_channel: 0.0,
            self._hand_joint_meas_channel: 0.0,
        }
        self._pub_dt = {
            self._arm_joint_meas_channel: 1.0 / arm_joint_meas_freq,
            self._hand_joint_meas_channel: 1.0 / hand_joint_meas_freq,
        }

    def set_mj_data_name_idx(self, mj_data: mj.MjData):
        for joint_name in self.arm_joint_name:
            self._list_of_arm_joint_indices.append(
                mj.mj_name2id(mj_data.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            )
        for joint_name in self.hand_joint_name:
            self._list_of_hand_joint_indices.append(
                mj.mj_name2id(mj_data.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            )
        self._list_of_joint_indices = (
            self._list_of_arm_joint_indices + self._list_of_hand_joint_indices
        )

    def apply_sim_control(self, ctrl_data: JointCtrlData, mj_data: mj.MjData):
        # Startup-only safety hold. Before the FIRST real controller command ever
        # arrives, fall back to the seeded hold + arm gravity comp: this env runs
        # with gravity_comp=False, so a raw zero/PD command does NOT hold the arm
        # (joint_2 alone needs ~56 Nm) and it would free-fall and blow up before
        # the controller connects.
        #
        # Once a controller has connected, we must NOT keep adding gravity comp on
        # empty-queue ticks: the held _last_ctrl_data already carries the
        # controller's OWN tau_ff = g(q), so adding qfrc_bias on top would be
        # double gravity comp and send the arm shooting upward. So the fallback
        # comp is gated on "never received a real command yet", not on this tick
        # being empty.
        if ctrl_data is None:
            ctrl_data = self._last_ctrl_data
        else:
            self._ever_received_ctrl = True

        _, arm_q, arm_qd, _ = self.arm_joint_meas_data.get_data()
        _, hand_q, hand_qd, _ = self.hand_joint_meas_data.get_data()
        q = np.concatenate((arm_q, hand_q))
        qd = np.concatenate((arm_qd, hand_qd))

        t, q_des, qd_des, tau_ff, kp, kd = ctrl_data.get_data()
        self._last_ctrl_data.set_data(t, q_des, qd_des, tau_ff, kp, kd)

        tau_command = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff

        if self._gravity_comp or not self._ever_received_ctrl:
            # Gravity/Coriolis feed-forward on the ARM only, matching SimFlexivArm
            # and the real Flexiv RT impedance modes (a zero command then holds
            # pose). The light hand is held by its PD gains, like the real hand
            # (which has no gravity comp).
            arm_slice = slice(0, _NUM_ARM_JOINTS)
            tau_command[arm_slice] = tau_command[arm_slice] + (
                mj_data.qfrc_bias[self._list_of_arm_joint_indices]
            )

        mj_data.ctrl[self._list_of_joint_indices] = tau_command

    def sync_intr_data_from_sim(self, mj_data: mj.MjData, intr_pub_que_dict: dict):
        t = mj_data.time

        # [Arm] accumulator-deadline gating so the long-term publish rate matches
        # pub_dt exactly (see SimFlexivArm for the rationale).
        arm_ch = self._arm_joint_meas_channel
        if t + 1e-9 >= self._last_pub_time[arm_ch] + self._pub_dt[arm_ch]:
            arm_q = mj_data.qpos[self._list_of_arm_joint_indices]
            arm_qd = mj_data.qvel[self._list_of_arm_joint_indices]
            arm_tau = mj_data.qfrc_actuator[self._list_of_arm_joint_indices]
            self.arm_joint_meas_data.set_data(t, arm_q, arm_qd, arm_tau)
            intr_pub_que_dict[arm_ch].put(copy.deepcopy(self.arm_joint_meas_data))
            self._last_pub_time[arm_ch] = (
                self._last_pub_time[arm_ch] + self._pub_dt[arm_ch]
            )

        # [Hand]
        hand_ch = self._hand_joint_meas_channel
        if t + 1e-9 >= self._last_pub_time[hand_ch] + self._pub_dt[hand_ch]:
            hand_q = mj_data.qpos[self._list_of_hand_joint_indices]
            hand_qd = mj_data.qvel[self._list_of_hand_joint_indices]
            hand_tau = mj_data.qfrc_actuator[self._list_of_hand_joint_indices]
            self.hand_joint_meas_data.set_data(t, hand_q, hand_qd, hand_tau)
            intr_pub_que_dict[hand_ch].put(copy.deepcopy(self.hand_joint_meas_data))
            self._last_pub_time[hand_ch] = (
                self._last_pub_time[hand_ch] + self._pub_dt[hand_ch]
            )

    def reset(self, mj_data: mj.MjData):
        # Seed the "last command" with a gentle hand hold (and a floating arm)
        # so the fingers stay put if the env runs before/without a controller
        # (ctrl_data is None -> this is applied). Without a hold the unactuated
        # fingers free-fall into their joint limits and the sim can blow up.
        # Any real controller command immediately overrides this.
        hold_kp = np.zeros(_NUM_JOINTS)
        hold_kd = np.zeros(_NUM_JOINTS)
        hold_kp[_NUM_ARM_JOINTS:] = self._default_hand_hold_kp
        hold_kd[_NUM_ARM_JOINTS:] = self._default_hand_hold_kd
        # Light arm PD to keep it centered at the default pose during the startup
        # gap. Gravity comp (added in apply_sim_control when no controller is
        # driving) carries the load; this just kills drift from that marginally-
        # stable equilibrium.
        hold_kp[:_NUM_ARM_JOINTS] = 40.0
        hold_kd[:_NUM_ARM_JOINTS] = 4.0

        # Re-arm the startup gravity-comp fallback for this run.
        self._ever_received_ctrl = False
        self._last_ctrl_data.set_data(
            mj_data.time,
            self._default_q,
            np.zeros(_NUM_JOINTS),
            np.zeros(_NUM_JOINTS),
            hold_kp,
            hold_kd,
        )
        self.arm_joint_meas_data.set_q(self._default_q[:_NUM_ARM_JOINTS])
        self.hand_joint_meas_data.set_q(self._default_q[_NUM_ARM_JOINTS:])

        mj_data.qpos[self._list_of_joint_indices] = self._default_q
        mj_data.qvel[self._list_of_joint_indices] = np.zeros((_NUM_JOINTS,))
        mj_data.qfrc_actuator[self._list_of_joint_indices] = np.zeros((_NUM_JOINTS,))
        mj_data.ctrl[self._list_of_joint_indices] = np.zeros((_NUM_JOINTS,))

        self._last_pub_time[self._arm_joint_meas_channel] = mj_data.time
        self._last_pub_time[self._hand_joint_meas_channel] = mj_data.time

    def get_col_info_data(self):
        """Robot collision primitives (per arm link + hand), in each link's
        local frame. Consumed by the env, published on robot_col_info, and used
        by the collision-aware controller for FCL distance queries."""
        return _build_arm_collision_primitives()
