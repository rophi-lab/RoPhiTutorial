from abc import ABC, abstractmethod

import lcm
import time
import threading
from queue import Queue

from communication.lcm.publisher.data_publisher.ClockPublisher import ClockPublisher


class BasePubManager(ABC):
    """
    Base class for the publisher manager.
    """

    def __init__(self, *args, max_pub_freq=2000, **kwargs):
        self.pub_que_dict = {}
        self.publisher_dict = {}
        self._lcm_instance = lcm.LCM()
        self._max_pub_freq = max_pub_freq
        self._pub_wait_dt = 1.0 / self._max_pub_freq
        self._stop_event = threading.Event()
        self._pub_thread = threading.Thread(target=self._run, daemon=False)

    def start(self):
        """
        Start the publisher manager.
        """
        # Create a thread to run the publisher manager
        # pub_thread = threading.Thread(target=self._run, daemon=True)
        self._pub_thread.start()

    def stop(self):
        """
        Stop the publisher manager.
        """
        self._stop_event.set()
        self._pub_thread.join()
        print("Publisher closed.")

    def _run(self):
        """
        Run the publisher manager.
        """
        try:
            while not self._stop_event.is_set():
                for pub in self.publisher_dict.values():
                    pub.check_que_and_publish()

                # Sleep to maintain the desired publishing frequency
                time.sleep(self._pub_wait_dt)
        except KeyboardInterrupt:
            pass

    def get_pub_que_dict(self):
        """
        get the publisher queue dictionary.
        """
        return self.pub_que_dict


class BaseEnvPubManager(BasePubManager):
    """
    Base class for the environment publisher manager.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseEnvPubManager.
        """
        super().__init__(*args, **kwargs)

        ## This is for env pub manager
        self.pub_que_dict["intr_pub_que_dict"] = {}
        self.pub_que_dict["extr_pub_que_dict"] = {}
        self.pub_que_dict["time_pub_que"] = Queue()

        self.clock_pub = ClockPublisher(
            self._lcm_instance,
            "sim_clock",
            self.pub_que_dict["time_pub_que"],
        )
        self.publisher_dict["sim_clock"] = self.clock_pub


class BaseCtrlPubManager(BasePubManager):
    """
    Base class for the controller publisher manager.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseCtrlPubManager.
        """
        super().__init__(*args, **kwargs)

        ## This is for ctrl pub manager
        self.ctrl_pub_que = Queue()
        self.pub_que_dict = {}

    def get_ctrl_pub_que(self):
        """
        get the ctrl publisher queue.
        """
        return self.ctrl_pub_que

    def get_pub_que_dict(self):
        """
        get the publisher queue dictionary.
        """
        return self.pub_que_dict
