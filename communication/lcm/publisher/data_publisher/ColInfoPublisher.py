from queue import Queue
import time

import numpy as np

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.ColInfoData import ColInfoData
from lcm_type.collision.col_info_t import col_info_t


class ColInfoPublisher(BaseDataPublisher):
    """
    ColInfoPublisher is a class that inherits from BaseDataPublisher.
    It is responsible for publishing collision information on a specific LCM channel.
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
            data = self._data_que.get()
            if isinstance(data, ColInfoData):
                t, dict_col_geoms = data.get_data()
                num_col_geoms = 0
                list_size_dims = []
                list_offsets = []
                concat_sizes = []
                concat_names = ""
                concat_geom_types = ""
                for name, dict_col_geom in dict_col_geoms.items():
                    num_col_geoms += 1
                    list_size_dims.append(int(dict_col_geom["size_dim"]))
                    list_offsets.append(
                        dict_col_geom["offset"].astype(np.float32).tolist()
                    )
                    concat_sizes.append(dict_col_geom["size"].astype(np.float32))
                    concat_names += name + ","
                    concat_geom_types += dict_col_geom["type"] + ","
                # remove the last comma
                concat_names = concat_names[:-1]
                concat_geom_types = concat_geom_types[:-1]

                concat_sizes = np.concatenate(concat_sizes, axis=0)
                len_concat_sizes = len(concat_sizes)

                msg = col_info_t()
                msg.timestamp = t
                msg.num_col_geoms = num_col_geoms
                msg.len_concat_sizes = len_concat_sizes
                msg.concat_names = concat_names
                msg.concat_geom_types = concat_geom_types
                msg.list_size_dims = list_size_dims  # (N,)
                msg.list_offsets = list_offsets  # (N, 4, 4)
                msg.concat_sizes = concat_sizes.tolist()  # (sum(size_dims),)

                self._lcm_instance.publish(self._channel, msg.encode())
