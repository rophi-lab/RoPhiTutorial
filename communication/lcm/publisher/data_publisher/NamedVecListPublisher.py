from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.NamedVecListData import NamedVecListData
from lcm_type.vector.vec_list_t import vec_list_t


class NamedVecListPublisher(BaseDataPublisher):
    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, NamedVecListData):
            return
        t, name_list, vec_list = data.get_data()
        msg = vec_list_t()
        msg.timestamp = float(t)
        msg.num_vecs = int(len(name_list))
        msg.vec_dim = int(data.get_vec_dim())
        msg.name_list = list(name_list)
        msg.vec_list = np_asarray_float(vec_list).tolist()
        self._lcm_instance.publish(self._channel, msg.encode())


def np_asarray_float(vec_list):
    import numpy as np

    return np.asarray(vec_list, dtype=np.float32)
