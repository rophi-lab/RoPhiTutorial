from abc import ABC, abstractmethod

import lcm


class BaseDataSubscriber(ABC):
    """
    Base class for LCM data subscribers.
    """

    def __init__(self, lcm_instance: lcm.LCM):
        """
        Initialize the BaseDataSubscriber with an LCM instance.

        @param[in]: lcm_instance (lcm.LCM): The LCM instance to use for subscribing to messages.
        """
        self._lcm_instance = lcm_instance

    def subscribe(self, channel: str):
        """
        Subscribe to the given channel.
        """
        self._lcm_instance.subscribe(channel, self._callback)

    @abstractmethod
    def _callback(self, channel: str, data):
        """
        Callback method to handle incoming messages.

        This method should be implemented by subclasses to process the received messages.
        """
        raise NotImplementedError("Subclasses should implement this method.")
