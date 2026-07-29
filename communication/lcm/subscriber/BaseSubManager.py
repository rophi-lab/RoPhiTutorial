from abc import ABC, abstractmethod

import lcm
from queue import Queue
import threading
import time


class BaseSubManager(ABC):
    """
    Base class for subscriber managers.
    """

    def __init__(self, *args, max_sub_freq=1000, **kwargs):
        """
        Initialize the BaseSubManager.
        """
        self._lcm_instance = (
            lcm.LCM()
        )  # this should be the only lcm instance for the whole subscriber
        self.data_queue_dict = {}
        self._max_sub_freq = max_sub_freq
        self._sub_wait_dt = 1 / self._max_sub_freq

        self._stop_event = threading.Event()
        self._sub_thread = threading.Thread(target=self._run, daemon=False)

    def start(self):
        """
        Start the subscriber manager.
        """
        # Create a thread to run the subscriber manager
        self._sub_thread.start()

    def stop(self):
        """
        Stop the subscriber manager.
        """
        self._stop_event.set()
        self._sub_thread.join()
        print("Subscriber closed.")

    def _run(self):
        """
        run the subscriber manager.
        """
        try:
            while not self._stop_event.is_set():
                self._lcm_instance.handle_timeout(int(1000 * self._sub_wait_dt))
        except KeyboardInterrupt:
            pass

    def get_sub_que_dict(self):
        """
        This method returns a dictionary of data queues for all the subscribed messages in this environment.
        return: a dictionary of all the data queues
        """
        return self.data_queue_dict


class BaseEnvSubManeager(BaseSubManager):
    """
    Base class for environment subscriber managers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseEnvSubManager.
        """
        super().__init__(*args, **kwargs)

        ## This is for env sub manager
        self.data_queue_dict["ctrl_sub_que"] = Queue()
        self.data_queue_dict["intr_sub_que_dict"] = {}
        self.data_queue_dict["extr_sub_que_dict"] = {}


class BaseControllerSubManager(BaseSubManager):
    """
    Base class for controller subscriber managers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseControllerSubManager.
        """
        super().__init__(*args, **kwargs)

        ## This is for controller sub manager
        self.data_queue_dict["intr_sub_que_dict"] = {}
        self.data_queue_dict["extr_sub_que_dict"] = {}
        self.data_queue_dict["percep_sub_que_dict"] = {}
        self.data_queue_dict["time_sub_que"] = Queue()


class BasePerceptionSubManager(BaseSubManager):
    """
    Base class for perception subscriber managers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BasePerceptionSubManager.
        """
        super().__init__(*args, **kwargs)

        ## This is for perception sub manager
        self.data_queue_dict["intr_sub_que_dict"] = {}
        self.data_queue_dict["extr_sub_que_dict"] = {}


class BaseSensorSubManager(BaseSubManager):
    """
    Base class for sensor subscriber managers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseSensorSubManager.
        """
        super().__init__(*args, **kwargs)

        ## This is for sensor sub manager
        self.data_queue_dict = {}
        # self.data_queue_dict["time_sub_que"] = Queue()
