from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.ImpGainsCmdData import ImpGainsCmdData
from lcm_type.ctrl.imp_gains_cmd_t import imp_gains_cmd_t


class ImpGainsCmdSubscriber(BaseDataSubscriber):
    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = imp_gains_cmd_t.decode(data)
        cmd = ImpGainsCmdData()
        cmd.set_data(
            float(msg.timestamp),
            float(msg.kq_scale),
            float(msg.dq_scale),
            float(msg.kt_trans_scale),
            float(msg.kt_rot_scale),
            float(msg.dt_trans_scale),
            float(msg.dt_rot_scale),
            mode=int(msg.mode),
        )
        self.data_queue.put(cmd)
