from abc import ABC, abstractmethod
from omegaconf import DictConfig
import select
import sys
import termios
import tty
import time
import lcm
import threading
from queue import Queue
import viser


class BaseVisManager(ABC):
    """
    Base class for the visualizer.
    """

    def __init__(self, config: DictConfig, max_sub_freq=1000, *args, **kwargs):
        """
        Initialize the visualizer.
        @param[in] cfg_visualizer: Configuration for the visualizer.
        """
        # This will be collected by the visualizer class
        # Each visualizer will have its own list of data subscribers
        self.data_queue_dict = {}
        self.data_queue_dict["ctrl_sub_que"] = Queue()
        self.data_queue_dict["intr_sub_que_dict"] = {}
        self.data_queue_dict["extr_sub_que_dict"] = {}
        self.data_queue_dict["percep_sub_que_dict"] = {}
        self.data_queue_dict["time_sub_que"] = Queue()

        self._lcm_instance = (
            lcm.LCM()
        )  # this should be the only lcm instance for the whole subscriber

        self._max_sub_freq = max_sub_freq
        self._sub_wait_dt = 1 / self._max_sub_freq

        self._stop_event = threading.Event()
        self._sub_thread = threading.Thread(target=self._run_sub_manager, daemon=False)

        self._vis_freq = config.get("vis_freq", 30)
        self._vis_dt = 1.0 / self._vis_freq
        self._last_vis_t = time.time()

        self._pause = False
        self._finished = False

        self.viser = viser.ViserServer()
        print("[Visualizer] Open your browser to http://localhost:8080")
        print("[Visualizer] Click the link. This will wait for 3 seconds.")
        time.sleep(3)

        self._set_visualizers(config)

    def __str__(self):
        """
        String representation of the visualizer.
        """
        return f"Visualizer: {self.__class__.__name__}"

    def initialize(self):
        """
        Initialize the visualizer.
        """
        pass

    def _set_visualizers(self, config: DictConfig):
        """
        Set the visualizers.
        """
        self._list_visualizers = []

    def start(self):
        """
        Start the visualizer.
        """
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self._sub_thread.start()

        try:
            while not self._finished:
                if self._keyboard_input_available():
                    key = sys.stdin.read(1)
                    self._handle_key(key)

                if not self._pause:
                    if time.time() - self._last_vis_t > self._vis_dt:
                        self._last_vis_t = time.time()
                        for vis in self._list_visualizers:
                            vis.update()
                # NOTE: this is to prevent the while loop from taking over the whole CPU
                time.sleep(0.0001)

        except KeyboardInterrupt:
            print("[Visualizer] Keyboard interrupt. Exiting.")
            self._finished = True
            self.stop()
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _run_sub_manager(self):
        """
        run the subscriber manager.
        """
        try:
            while not self._stop_event.is_set():
                self._lcm_instance.handle_timeout(int(1000 * self._sub_wait_dt))
        except KeyboardInterrupt:
            pass

    def stop(self):
        """
        Stop the visualizer.
        """
        self._stop_event.set()
        if self._sub_thread.is_alive():
            self._sub_thread.join()
        print("[Visualizer] Visualizer stopped.")

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
            print(f"[Visualizer] Toggled pause: {self._pause}")
        elif key == "q":
            print("[Visualizer] Quit signal received.")
            self._finished = True
        else:
            print(f"[Visualizer] Key pressed: {key}")
