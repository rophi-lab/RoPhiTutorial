import os
import sys
import copy
import time
import signal
import atexit
import subprocess

import numpy as np
from omegaconf import DictConfig

from robot_platform.hardware.BaseHardwarePlatform import BaseHardwarePlatform
from robot_platform.hardware.brl_arm_5F_hand.Robotis5FHardware import Robotis5FHardware
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData
from utils.mode import HandControlMode


_NUM_ARM_JOINTS = 7
_NUM_HAND_JOINTS = 20
_NUM_JOINTS = _NUM_ARM_JOINTS + _NUM_HAND_JOINTS


class HardwareFlexivArm5FHand(BaseHardwarePlatform):
    """Combined Flexiv Rizon4 arm + Robotis RH-5 hand hardware platform.

    Split by how each subsystem is actuated:

    - **Arm**: handled entirely by the C++ Flexiv bridge, spawned here as a
      subprocess (exactly like HardwareFlexivArm). The bridge does its own LCM:
      it subscribes to the unified command channel, slices the arm portion
      (mapping.ctrl_joint_offset=0, num_arm_joints=7), computes torque, streams
      it to the robot, and publishes the arm joint_meas. When it gets no fresh
      command it idles the robot (safe hold).

    - **Hand**: driven here in Python via Robotis5FHardware over serial. We read
      the same unified 27-DoF command off ``ctrl_sub_que``, use the hand slice
      [7:27], command the hand, read its state, and publish the hand joint_meas.

    So the controller publishes ONE 27-DoF command ([arm(7), hand(20)]) and the
    bridge + this platform each take their slice - matching the sim's contract.

    ``read_only`` (dry-run) mode: the hand is read and published but never
    actuated (torque disabled), and commands are ignored. Combined with a bridge
    config whose ctrl channel nothing publishes to, this makes the whole robot a
    pure sensor - used by the sim-shadow torque-verification harness.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._read_only = bool(config.get("read_only", False))

        # ---------------- Arm (bridge subprocess) ----------------
        arm_cfg = config["arm"]
        self._bridge_binary = arm_cfg["bridge_binary"]
        self._bridge_config = arm_cfg["bridge_config"]
        self._bridge_ld_lib_path = os.path.expanduser(
            arm_cfg.get("bridge_ld_library_path", "~/rdk_install/lib")
        )
        self._bridge_proc = None

        # ---------------- Hand (serial) ----------------
        hand_cfg = config["hand"]
        hand_mode_str = hand_cfg.get("control_mode", "CURRENT_CONTROL")
        try:
            self._hand_control_mode = HandControlMode[hand_mode_str]
        except KeyError:
            raise ValueError(f"[Hardware] Invalid hand control mode: {hand_mode_str}")

        self._hand_devicename = hand_cfg["devicename"]
        self._hand_zero_pos = np.array(hand_cfg["zero_pos"])
        self._hand_hardware = Robotis5FHardware(
            self._hand_devicename, self._hand_zero_pos, self._hand_control_mode
        )

        self._hand_default_q = np.array(
            hand_cfg.get("default_q", np.zeros(_NUM_HAND_JOINTS))
        )
        self._hand_hold_kp = np.array(
            hand_cfg.get("hold_kp", [0.3] * _NUM_HAND_JOINTS)
        )
        self._hand_hold_kd = np.array(
            hand_cfg.get("hold_kd", [0.03] * _NUM_HAND_JOINTS)
        )
        self._hand_joint_state = JointMeasData(num_joints=_NUM_HAND_JOINTS)
        self._hand_joint_meas_channel = config["pub_manager"][
            "hand_joint_meas_channel"
        ]

        # ---------------- Command bookkeeping ----------------
        self._last_ctrl_data = JointCtrlData(num_joints=_NUM_JOINTS)
        self._received_first_ctrl = False
        self._last_ctrl_t = time.time()
        self._ctrl_timeout_t = float(config.get("ctrl_timeout", 0.5))
        # Rate-limit hand command/publish to hardware_freq (self._update_dt).
        # _step() is called by the base loop as fast as it can spin, so without
        # this the hand meas floods the publisher's queue (which drains one
        # sample per cycle) and readers see rapidly growing latency.
        self._last_hand_t = time.time()

        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # run_hardware.py wiring
    # ------------------------------------------------------------------

    def set_sub_que_dict(self, sub_data_queue_dict):
        # We DO need the command queue (for the hand slice); the bridge takes
        # the arm slice independently off its own LCM subscription.
        self.ctrl_sub_que = sub_data_queue_dict["ctrl_sub_que"]

    def set_pub_que_dict(self, pub_data_queue_dict):
        self.intr_pub_que_dict = pub_data_queue_dict["intr_pub_que_dict"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self):
        print("[Hardware] Starting Robotis hand...")
        self._hand_hardware.start()
        if self._read_only:
            # Dry-run: leave the hand un-commanded. In CURRENT_CONTROL, with no
            # command() call the IO thread never writes a goal current, so the
            # fingers stay back-drivable (~0 torque) while still being READ.
            # We deliberately do NOT call disable_torque() here: it writes to the
            # serial bus while the hand's IO thread is already reading/writing it
            # (they don't share a port lock), which corrupts the state reads and
            # freezes the reported finger positions.
            print("[Hardware] read_only: hand left un-commanded (sensor-only).")

        print("[Hardware] Spawning Flexiv bridge...")
        self._spawn_bridge()

        self._hardware_ready = True
        print(
            f"[Hardware] Ready (bridge pid={self._bridge_proc.pid}"
            f"{', READ_ONLY' if self._read_only else ''}). "
            f"Press 'q' (or Ctrl-C) to shut down."
        )

    def start(self):
        try:
            super().start()
        finally:
            self.shutdown()

    def shutdown(self):
        # Bring the bridge down first so Flexiv's own safety takes over the arm.
        if self._bridge_proc is not None and self._bridge_proc.poll() is None:
            print("[Hardware] Sending SIGINT to bridge...")
            try:
                self._bridge_proc.send_signal(signal.SIGINT)
                self._bridge_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[Hardware] Bridge didn't exit; SIGTERM.")
                self._bridge_proc.terminate()
                try:
                    self._bridge_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print("[Hardware] Bridge still alive; SIGKILL.")
                    self._bridge_proc.kill()
        self._bridge_proc = None

        try:
            self._hand_hardware.stop()
            print("[Hardware] Hand stopped.")
        except Exception as e:
            print(f"[Hardware] Error stopping hand: {e}")

    def estop(self):
        # Full shutdown: bridge down (Flexiv safety holds the arm), hand off.
        print("[Hardware] ESTOP.")
        self._finished = True
        self.shutdown()

    # ------------------------------------------------------------------
    # Main-loop hooks
    # ------------------------------------------------------------------

    def _step(self):
        # 1) Watch the bridge subprocess.
        if self._bridge_proc is not None and self._bridge_proc.poll() is not None:
            print(
                f"[Hardware] Bridge exited unexpectedly "
                f"(rc={self._bridge_proc.returncode}); shutting down."
            )
            self._finished = True
            return

        # 2) Drain the latest unified command.
        if not self.ctrl_sub_que.empty():
            while not self.ctrl_sub_que.empty():
                self._last_ctrl_data = self.ctrl_sub_que.get()
            self._received_first_ctrl = True
            self._last_ctrl_t = time.time()

        # Rate-limit the hand command + publish to hardware_freq. (Draining the
        # command queue above stays every-tick so the latest command is used.)
        if time.time() - self._last_hand_t < self._update_dt:
            return
        self._last_hand_t = time.time()

        # 3) Command the hand (skipped entirely in read-only/dry-run).
        if not self._read_only:
            fresh = self._received_first_ctrl and (
                time.time() - self._last_ctrl_t < self._ctrl_timeout_t
            )
            if fresh:
                _, q_des, qd_des, tau_ff, kp, kd = self._last_ctrl_data.get_data()
                self._hand_hardware.command(
                    q_des[_NUM_ARM_JOINTS:],
                    qd_des[_NUM_ARM_JOINTS:],
                    tau_ff[_NUM_ARM_JOINTS:],
                    kp[_NUM_ARM_JOINTS:],
                    kd[_NUM_ARM_JOINTS:],
                )
            else:
                # No fresh command: hold the default hand pose gently.
                self._hand_hardware.command(
                    self._hand_default_q,
                    np.zeros(_NUM_HAND_JOINTS),
                    np.zeros(_NUM_HAND_JOINTS),
                    self._hand_hold_kp,
                    self._hand_hold_kd,
                )

        # 4) Read + publish the hand state. (Arm meas is published by the bridge.)
        q, qd, tau = self._hand_hardware.get_motor_data()
        self._hand_joint_state.set_data(time.time(), q, qd, tau)
        self.intr_pub_que_dict[self._hand_joint_meas_channel].put(
            copy.deepcopy(self._hand_joint_state)
        )

    def _handle_key(self, key):
        if key == "q":
            print("[Hardware] Quit requested.")
            self._finished = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spawn_bridge(self) -> None:
        env = os.environ.copy()
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            self._bridge_ld_lib_path + (":" + existing if existing else "")
        )
        self._bridge_proc = subprocess.Popen(
            [self._bridge_binary, self._bridge_config],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
