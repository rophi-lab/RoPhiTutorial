import sys
import select
import time
import termios
import tty

from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseHardwarePlatform(ABC):
    """
    Base class for all hardware platforms.
    This class defines several methods for applying control and getting data from the hardware.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):

        self._hardware_ready = False

        self._finished = False

        self._freq = float(config.get("hardware_freq", 1000.0))
        self._update_dt = 1.0 / self._freq

        # These are initialized in the publish/subscribe queue dictionary
        self.ctrl_sub_que = None
        self.intr_pub_que_dict = {}
        self.time_pub_que = None  # queue for publishing time

        self._last_clock_pub_time = time.time()

    def __str__(self):
        return f"Hardware Platform: {self.__class__.__name__}"

    @abstractmethod
    def initialize(self):
        """
        Initialize the hardware platform.
        """
        raise NotImplementedError(
            "[Hardware] initialize method must be implemented in subclasses."
        )

    def start(self):
        """
        Start the hardware platform.
        """
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while not self._finished:
                if self._keyboard_input_available():
                    key = sys.stdin.read(1)
                    self._handle_key(key)

                if self._hardware_ready:
                    self._step()

                # put the current time in the time queue
                # for clock synchronization
                if time.time() - self._last_clock_pub_time > self._update_dt:
                    self._last_clock_pub_time = time.time()
                    self._publish_time()

                # NOTE: this is to prevent the while loop from taking over the whole CPU
                time.sleep(0.0001)

        except KeyboardInterrupt:
            print("[Hardware] Keyboard interrupt. Exiting.")

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def reset(self):
        """
        Reset the hardware platform to its initial state.
        """
        pass

    @abstractmethod
    def shutdown(self):
        """
        Stop the hardware platform.
        """
        raise NotImplementedError(
            "[Hardware] shutdown method must be implemented in subclasses."
        )

    @abstractmethod
    def estop(self):
        """
        Emergency stop the hardware platform.
        """
        raise NotImplementedError(
            "[Hardware] estop method must be implemented in subclasses."
        )

    def set_sub_que_dict(self, sub_data_queue_dict):
        """
        Set the control subscription queue.
        @param[in] ctrl_sub_que: The control subscription queue to set.
        """
        self.ctrl_sub_que = sub_data_queue_dict["ctrl_sub_que"]

    def set_pub_que_dict(self, pub_data_queue_dict):
        """
        Set the data queue for the hardware platform.
        @param[in] data_queue_dict: The data queue dictionary to set.
        """
        self.intr_pub_que_dict = pub_data_queue_dict["intr_pub_que_dict"]

    def _keyboard_input_available(self):
        """
        Check if a keyboard input is available (non-blocking).
        """
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def _handle_key(self, key):
        """
        Handle keyboard input and change internal variables accordingly.
        """
        pass

    def _publish_time(self):
        """
        Publish the current time to the time queue.
        """
        if self.time_pub_que is not None:
            t = time.time()
            self.time_pub_que.put(t)

    @abstractmethod
    def _step(self):
        """
        Apply control data to the platform.
        @param[in] ctrl_data: Control data to apply.
        """
        raise NotImplementedError(
            "apply_control method must be implemented in subclasses."
        )
