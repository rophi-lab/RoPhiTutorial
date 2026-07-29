# import termios
import time
from queue import Queue

from omegaconf import DictConfig

from utils.mode import EnvMode

import os
import sys
import select
import termios
import tty


class BaseController:
    def __init__(self, config: DictConfig, *args, **kwargs):
        self._ctrl_freq = config["ctrl_freq"]

        env_mode_cfg = config.get("env_mode", "sim_real_time")
        if env_mode_cfg == "sim_real_time":
            # in sim_real_time mode the step function will wait until the simulation time matches the real time.
            # (The simulation will not run faster than real time).
            self.env_mode = EnvMode.SIM_REAL_TIME
        elif env_mode_cfg == "sim_sim_time":
            # in sim_sim_time mode the step function will run as fast as possible.
            self.env_mode = EnvMode.SIM_SIM_TIME
        elif env_mode_cfg == "real":
            self.env_mode = EnvMode.REAL
        else:
            raise ValueError(
                f"Invalid environment mode: {env_mode_cfg}. \
                    Must be 'sim_real_time', 'sim_sim_time' or 'real'."
            )

        self.intr_sub_que_dict = None
        self.extr_sub_que_dict = None
        self.percep_sub_que_dict = None
        self.time_sub_que = None

        self.pub_que_dict = None
        self.ctrl_pub_que = None

        # Current time for the system and measurements. in both simulation mode this will be sim_time, in real mode this will be real time.
        self._cur_time = 0
        self._last_ctrl_t = 0

        # The current time for making sure the controller follows the control frequency.
        # in sim_real_time mode this will be sim_time.
        # in both sim_sim_time and real mode this will be real time.
        # In sim_sim_time, this will help ensure the controller does not run fater than simulation step process time.
        # self._freq_check_cur_ctrl_t = 0
        # This is the same unit as _freq_check_cur_ctrl_t, but record the last time the controller was called.
        # self._freq_check_last_ctrl_t = 0

        self._ctrl_dt = 1.0 / self._ctrl_freq
        self._pause = False
        self._finished = False

        # NOTE: Necessary for sim_sim_time mode.
        # This is set True, so that when the controller is started,
        # no matter env is running or not
        # (i.e. whether the controller received the sim time or not),
        # the controller will initiate the sim and controller loop.
        self._sim_time_received = True

    def __str__(self):
        return (
            f"Controller: {self.__class__.__name__}, " f"ctrl_freq: {self._ctrl_freq}, "
        )

    def initialize(self):
        """
        Initialize the controller. This function should be overridden by the child class.
        """
        # NOTE: The user should implement the initialization of the controller
        # in the child class.
        pass

    def reset(self):
        """
        Reset the controller.
        """
        self._finished = False
        self._pause = False
        self._sim_time_received = False
        self._last_ctrl_t = self._cur_time
        print("[Controller] Controller reset complete.")

    def start(self):
        """
        Main loop of the controller.
        """

        self._sub_thread_start()

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        # Seed the control-rate deadline from the current time once before the
        # loop starts. Without this, _last_ctrl_t is 0 from __init__ and the
        # catch-up branch fires every iteration forever (loop runs at iteration
        # rate, not _ctrl_freq).
        self._update_time()
        self._last_ctrl_t = self._cur_time

        try:
            while not self._finished:
                if self._keyboard_input_available():
                    # Read straight from the fd (not sys.stdin.read, whose
                    # buffered stream reads ahead and stalls select(), stranding
                    # the rest of a multi-byte escape sequence and desyncing
                    # subsequent keys). One os.read delivers the whole sequence.
                    for key in os.read(sys.stdin.fileno(), 32).decode(errors="ignore"):
                        self._handle_key(key)

                if not self._pause:
                    self._update_time()
                    self._check_and_get_data_from_que()

                    if self.env_mode == EnvMode.SIM_SIM_TIME:
                        # In sim_sim_time mode, the controller will run as fast as possible,
                        # but we need to check if the simulation time has been received.
                        # The control freq is controlled by env side.
                        if self._sim_time_received:
                            self._update()
                            self._sim_time_received = False
                    else:
                        if self._cur_time - self._last_ctrl_t > self._ctrl_dt:
                            # Advance the deadline by exactly _ctrl_dt rather
                            # than snapping to _cur_time, otherwise the ~100us
                            # sleep granularity below drifts the effective rate
                            # downward (e.g. 1000Hz -> ~900Hz).
                            self._last_ctrl_t += self._ctrl_dt
                            self._update()

                    # NOTE: We need this only for sim_sim_time mode.
                    # Otherwise the while loop runs too fast.
                    # Occupying the CPU.
                    time.sleep(0.0001)
        except KeyboardInterrupt:
            print("[Controller] Keyboard interrupt received. Stopping controller.")
            self._finished = True

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def stop(self):
        """
        Stop the controller.
        """
        self._finished = True
        print("[Controller] Controller stopped.")
        self._sub_thread_stop()

    def is_finished(self):
        """
        Check if the controller is finished.
        """
        return self._finished

    def set_sub_que_dict(self, sub_que_dict: dict):
        """
        Set the subscriber queue dictionary.
        @param[in] sub_que_dict: Subscriber queue dictionary.
        """
        self.intr_sub_que_dict = sub_que_dict["intr_sub_que_dict"]
        self.extr_sub_que_dict = sub_que_dict["extr_sub_que_dict"]
        self.percep_sub_que_dict = sub_que_dict["percep_sub_que_dict"]
        self.time_sub_que = sub_que_dict["time_sub_que"]

    def set_ctrl_pub_que(self, pub_que: Queue):
        """
        Set the control publisher queue.
        @param[in] pub_que: Publisher queue.
        """
        self.ctrl_pub_que = pub_que

    def set_pub_que_dict(self, pub_que_dict: dict):
        """
        Set the publisher queue dictionary.
        @param[in] pub_que_dict: Publisher queue dictionary.
        """
        self.pub_que_dict = pub_que_dict

    def _keyboard_input_available(self):
        """
        Check if a keyboard input is available (non-blocking).
        """
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def _handle_key(self, key):
        """
        Handle keyboard input and change internal variables accordingly.
        """
        if key == "p":
            self._pause = not self._pause
            print(f"[Controller] Toggled pause: {self._pause}")
        elif key == "q":
            print("[Controller] Quit signal received.")
            self._finished = True
        else:
            print(f"[Controller] Key pressed: {key}")

    def _update_time(self):
        """
        Update the current time.
        """
        if self.env_mode == EnvMode.SIM_REAL_TIME:
            # Drain to the latest stamp so a backlog (e.g. while RRT held the
            # GIL) does not make traj time chase stale / jumped samples one-by-one.
            if not self.time_sub_que.empty():
                t = self.time_sub_que.get()
                while not self.time_sub_que.empty():
                    t = self.time_sub_que.get()
                self._cur_time = t
        elif self.env_mode == EnvMode.SIM_SIM_TIME:
            # In sim_sim_time mode, the controller will run as fast as possible.
            if not self.time_sub_que.empty():
                self._sim_time_received = True
                self._cur_time = self.time_sub_que.get()
            # self._freq_check_cur_ctrl_t = time.time()
        elif self.env_mode == EnvMode.REAL:
            self._cur_time = time.time()

    def _check_and_get_data_from_que(self):
        """ """
        raise NotImplementedError(
            "The _check_and_get_data_from_que function should be overridden."
        )

    def _update(self):
        """
        Update the controller. This function should be overridden by the child class.

        This function check the intr_que, extr_que and percep_que, and produce the control
        command. This function should also put the control command into the ctrl_pub_que
        for the publisher.
        """
        raise NotImplementedError("The _update function should be overridden.")

    def _sub_thread_start(self):
        """
        Start the sub thread.
        """
        pass

    def _sub_thread_stop(self):
        """
        Stop the sub thread.
        """
        pass
