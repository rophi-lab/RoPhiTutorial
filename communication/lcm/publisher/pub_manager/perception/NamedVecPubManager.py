from queue import Queue

import lcm

from communication.lcm.publisher.BasePubManager import BasePubManager
from communication.lcm.publisher.data_publisher.NamedVecListPublisher import (
    NamedVecListPublisher,
)


class NamedVecPubManager(BasePubManager):
    """
    Class for managing the publisher for the BRL Arm Hand with one camera.
    """

    def __init__(self, *args, named_vec_list_channel: str = "named_vec_list", **kwargs):
        super().__init__(*args, **kwargs)

        self.pub_que_dict[named_vec_list_channel] = Queue()

        self.publisher_dict[named_vec_list_channel] = NamedVecListPublisher(
            self._lcm_instance,
            named_vec_list_channel,
            self.pub_que_dict[named_vec_list_channel],
        )
