from queue import Queue

import lcm
import numpy as np

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.NamedVecListData import NamedVecListData
from lcm_type.vector.vec_list_t import vec_list_t


class NamedVecListSubscriber(BaseDataSubscriber):
    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = vec_list_t.decode(data)
        named = NamedVecListData(num_vecs=msg.num_vecs, vec_dim=msg.vec_dim)
        named.set_data(msg.timestamp, list(msg.name_list), np.asarray(msg.vec_list))
        self.data_queue.put(named)
