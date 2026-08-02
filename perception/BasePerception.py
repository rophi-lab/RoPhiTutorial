from abc import ABC, abstractmethod

import time

from collections import defaultdict
from queue import Queue
from omegaconf import DictConfig


class BasePerception(ABC):
    """
    Base class for perception modules.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the base perception module.
        """

        self.intr_sub_que_dict = defaultdict(Queue)
        self.extr_sub_que_dict = defaultdict(Queue)
        self.time_sub_que = None
        self.percep_pub_que_dict = defaultdict(Queue)

        self._cur_time = 0

        self._finished = False
        self._pause = False

    def __str__(self):
        """
        Return a string representation of the perception module.
        """
        return f"Perception: {self.__class__.__name__}"

    def reset(self):
        """
        Reset the perception module.
        """
        self._finished = False
        self._pause = False
        print("[Perception] Perception reset complete.")

    def initialize(self):
        """
        Initialize the perception module.
        """
        # NOTE: The user should implement the initialization of the perception module
        # in the child class.
        pass

    def stop(self):
        """
        Stop the perception module.
        """
        self._finished = True
        print("[Perception] Perception stopped.")

    def start(self):
        """
        Start the perception module.
        """
        try:
            while not self._finished:
                if not self._pause:

                    # The process function should check the subscribing queue,
                    # process the perception data, and put the result in the publishing queue.
                    # NOTE: The user should implement the data synchronization within
                    # this function in the child class.
                    self._process()

                    # NOTE: We need this only for sim_sim_time mode.
                    # Otherwise the while loop runs too fast.
                    # Occupying the CPU.
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print(
                "[Perception] Keyboard interrupt received. Stopping the perception module."
            )
            self._finished = True

    def set_sub_que_dict(self, sub_que_dict: dict):
        """
        Set the subscriber queue dictionary.
        Args:
            sub_que_dict: Subscriber queue dictionary.
        """
        self.intr_sub_que_dict = sub_que_dict["intr_sub_que_dict"]
        self.extr_sub_que_dict = sub_que_dict["extr_sub_que_dict"]

    def set_percep_pub_que_dict(self, pub_que_dict: dict):
        """
        Set the publisher queue dictionary.
        Args:
            pub_que: Publisher queue dictionary.
        """
        self.percep_pub_que_dict = pub_que_dict

    @abstractmethod
    def _process(self):
        """
        Process the input data.

        Args:
            data: Input data to be processed.

        Returns:
            Processed data.
        """
        raise NotImplementedError("Subclasses should implement this method.")
