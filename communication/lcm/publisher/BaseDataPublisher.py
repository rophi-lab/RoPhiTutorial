from abc import ABC, abstractmethod

from queue import Queue

import lcm


class BaseDataPublisher(ABC):
    """
    Base class for LCM data publishers.
    """

    def __init__(self, lcm_instance: lcm.LCM, channel: str, data_que: Queue):
        """
        Initializes the LCM data publisher.

        @param[in]: lcm_instance: An instance of the LCM class.
        """
        self._lcm_instance = lcm_instance
        self._channel = channel
        self._data_que = data_que

    @abstractmethod
    def check_que_and_publish(self):
        """
        Publishes data on the specified LCM channel.

        @param[in]: channel: The LCM channel to publish data on.
        """
        raise NotImplementedError("Subclasses must implement this method.")
