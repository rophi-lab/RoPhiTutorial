from queue import Queue

import numpy as np

import lcm
from lcm_type.clock import clock_t

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber


class ClockSubscriber(BaseDataSubscriber):
    """
    ClockSubscriber class for subscribing to clock data from LCM channels.
    """

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        """
        Callback function for the subscriber.
        """
        msg = clock_t.decode(data)
        self.data_queue.put(msg.timestamp)
