"""Base environment: sim/real main loop, queues, and keyboard control."""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from abc import ABC, abstractmethod
from queue import Queue

from omegaconf import DictConfig

from robot_platform.sim import get_sim_platform
from utils.mode import EnvMode, VisMode


class BaseEnv(ABC):
    """Shared environment loop for simulation and hardware modes."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        """
        Args:
            config: OmegaConf with ``env_mode``, ``vis_mode``, ``platform``,
                ``sub_manager``, ``pub_manager``, and timing fields.
        """
        env_mode = config.get("env_mode", "sim_real_time")
        vis_mode = config.get("vis_mode", "vis_on")

        if env_mode == "sim_real_time":
            # Cap sim rate to wall clock; always step even without ctrl.
            self.env_mode = EnvMode.SIM_REAL_TIME
            self._sim_wait_control = False
            self._sim_steps_per_control = 1
        elif env_mode == "sim_sim_time":
            # Run as fast as possible; wait for control when enabled.
            self.env_mode = EnvMode.SIM_SIM_TIME
            self._sim_wait_control = True
            self._sim_steps_per_control = config.get("sim_steps_per_control", 1)
        elif env_mode == "real":
            self.env_mode = EnvMode.REAL
        else:
            raise ValueError(
                f"Invalid env_mode: {env_mode}. "
                "Expected 'sim_real_time', 'sim_sim_time', or 'real'."
            )

        if vis_mode == "vis_on":
            self.vis_mode = VisMode.ON
        elif vis_mode == "vis_off":
            self.vis_mode = VisMode.OFF
        else:
            raise ValueError(
                f"Invalid vis_mode: {vis_mode}. Expected 'vis_on' or 'vis_off'."
            )

        self._sim_freq = config.get("sim_freq", 1000)
        self._view_freq = config.get("view_freq", 30)
        self._sim_dt = 1.0 / self._sim_freq
        self._view_dt = 1.0 / self._view_freq

        self.platform = get_sim_platform(
            config["platform"], config["sub_manager"], config["pub_manager"]
        )

        # Wired later via set_sub_que_dict / set_pub_que_dict.
        self.ctrl_sub_que = None
        self.intr_sub_que_dict = None
        self.extr_sub_que_dict = None
        self.intr_pub_que_dict = None
        self.extr_pub_que_dict = None
        self.time_pub_que = None

        self._pause = False
        self._finished = False
        self._last_real_t = time.time()
        self._last_view_t = time.time()
        # Forces a clock publish right after unpausing so controllers wake up.
        self._last_pause = True

    def __str__(self) -> str:
        return (
            f"Environment: {self.__class__.__name__}, "
            f"mode: {self.env_mode}, vis: {self.vis_mode}, "
            f"platform: {self.platform}"
        )

    def reset(self) -> None:
        """Reset pause/finish flags and the platform."""
        self._pause = False
        self._finished = False
        self._last_real_t = time.time()
        self._last_view_t = time.time()
        self.platform.reset()
        print("[Env] Environment reset complete.")

    def initialize(self) -> None:
        """Optional hook before ``start()``."""

    def start(self) -> None:
        """Run the main environment loop until quit or viewer close."""
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        now = time.time()
        self._last_real_t = now
        self._last_view_t = now

        try:
            while not self._finished:
                if self._keyboard_input_available():
                    self._handle_key(sys.stdin.read(1))

                if self._pause:
                    self._last_pause = True
                    continue

                if self.env_mode == EnvMode.SIM_REAL_TIME:
                    if time.time() - self._last_real_t > self._sim_dt:
                        # Advance by fixed dt to avoid sleep drift.
                        self._last_real_t += self._sim_dt
                        self._step()
                else:
                    self._step()

                if self.vis_mode == VisMode.ON:
                    if time.time() - self._last_view_t > self._view_dt:
                        self._last_view_t = time.time()
                        self._update_vis()

                time.sleep(0.0001)
                self._check_stopping_condition()
                self._last_pause = False

        except KeyboardInterrupt:
            print("[Env] Keyboard interrupt. Exiting.")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def stop(self) -> None:
        """Optional cleanup after the main loop."""

    def is_finished(self) -> bool:
        """Return True when the environment loop should exit."""
        return self._finished

    def set_sub_que_dict(self, que_dict: dict) -> None:
        """Attach subscriber queues (control, and optionally hardware state)."""
        self.ctrl_sub_que = que_dict["ctrl_sub_que"]
        if self.env_mode == EnvMode.REAL:
            self.intr_sub_que_dict = que_dict["intr_sub_que_dict"]

    def set_pub_que_dict(self, que_dict: dict) -> None:
        """Attach publisher queues for measurements and sim clock."""
        self.intr_pub_que_dict = que_dict["intr_pub_que_dict"]
        self.extr_pub_que_dict = que_dict["extr_pub_que_dict"]
        self.time_pub_que = que_dict["time_pub_que"]

    def _keyboard_input_available(self) -> bool:
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def _handle_key(self, key: str) -> None:
        if key == "p":
            self._pause = not self._pause
            print(f"[Env] Toggled pause: {self._pause}")
        elif key == "q":
            print("[Env] Quit signal received.")
            self._finished = True
        elif key == "r":
            print("[Env] Resetting environment.")
            self.reset()
        else:
            print(f"[Env] Key pressed: {key}")

    def _step(self) -> None:
        """One control/sim tick (or hardware sync in REAL mode)."""
        if self.env_mode in (EnvMode.SIM_REAL_TIME, EnvMode.SIM_SIM_TIME):
            ctrl_data = None
            received = False
            while not self.ctrl_sub_que.empty():
                ctrl_data = self.ctrl_sub_que.get_nowait()
                received = True

            if received or not self._sim_wait_control:
                self._apply_sim_control(ctrl_data)
                self._step_simulation()
                self._update_sim_env()
                self._publish_sim_time(self.time_pub_que)
            elif self._last_pause:
                self._publish_sim_time(self.time_pub_que)

            self._sync_data_from_sim(
                self.intr_pub_que_dict,
                self.extr_pub_que_dict,
                self.time_pub_que,
            )
            return

        if self.env_mode == EnvMode.REAL:
            self._sync_data_from_hardware(
                self.ctrl_sub_que,
                self.intr_sub_que_dict,
                self.extr_sub_que_dict,
                self.intr_pub_que_dict,
                self.extr_pub_que_dict,
                self.time_pub_que,
            )

    def _update_sim_env(self) -> None:
        """Hook for time-varying scenes. Default: no-op."""

    @abstractmethod
    def _check_stopping_condition(self) -> None:
        """Set ``_finished`` when the run should stop."""

    @abstractmethod
    def _step_simulation(self) -> None:
        """Advance the physics simulator."""

    @abstractmethod
    def _apply_sim_control(self, ctrl_data) -> None:
        """Write control into the simulator."""

    @abstractmethod
    def _publish_sim_time(self, time_pub_que: Queue) -> None:
        """Publish simulation clock."""

    @abstractmethod
    def _sync_data_from_sim(
        self,
        intr_pub_que_dict: dict,
        extr_pub_que_dict: dict,
        time_pub_que: Queue,
    ) -> None:
        """Publish observations from the simulator."""

    @abstractmethod
    def _sync_data_from_hardware(
        self,
        ctrl_sub_que: Queue,
        intr_sub_que_dict: dict,
        extr_sub_que_dict: dict,
        intr_pub_que_dict: dict,
        extr_pub_que_dict: dict,
        time_pub_que,
    ) -> None:
        """Sync / republish hardware state (REAL mode)."""

    @abstractmethod
    def _update_vis(self) -> None:
        """Refresh the visualizer."""
