from queue import Queue

import numpy as np

import lcm
from lcm_type.collision.col_info_t import col_info_t

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.ColInfoData import ColInfoData


class ColInfoSubscriber(BaseDataSubscriber):
    """
    ColInfoSubscriber class for subscribing to collision information from LCM channels.
    """

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        """
        Callback function for the subscriber.
        """
        msg = col_info_t.decode(data)

        col_info_data = ColInfoData(name=channel)
        t = msg.timestamp
        num_col_geoms = msg.num_col_geoms
        len_concat_sizes = msg.len_concat_sizes
        concat_names = msg.concat_names
        concat_geom_types = msg.concat_geom_types
        list_size_dims = msg.list_size_dims
        concat_sizes = msg.concat_sizes
        list_offsets = msg.list_offsets

        list_names = concat_names.split(",")
        list_geom_types = concat_geom_types.split(",")

        col_info_data.set_time(t)

        for i in range(num_col_geoms):
            size_dim = list_size_dims[i]
            prev_size_dim_sum = sum(list_size_dims[:i])
            size = np.array(
                concat_sizes[prev_size_dim_sum : prev_size_dim_sum + size_dim],
            )
            col_info_data.add_col_geom(
                name=list_names[i],
                geom_type=list_geom_types[i],
                size=size,
                offset=np.array(list_offsets[i], dtype=np.float32),
            )

        self.data_queue.put(col_info_data)
