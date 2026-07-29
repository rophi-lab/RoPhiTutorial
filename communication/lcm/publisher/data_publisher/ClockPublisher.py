from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from lcm_type.clock import clock_t


class ClockPublisher(BaseDataPublisher):
    """
    Publish clock for time synchronization.
    """

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        """
        Check if the queue is not empty and publish the data to the LCM channel.
        """
        if not self._data_que.empty():
            t = self._data_que.get()
            if isinstance(t, float):
                # convert the data to the LCM message type
                msg = clock_t()
                msg.timestamp = t
                # publish the message to the LCM channel
                self._lcm_instance.publish(self._channel, msg.encode())
            else:
                print("Data is not valid, not publishing.")
