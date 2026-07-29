from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.P2PGainsCmdData import P2PGainsCmdData
from lcm_type.ctrl.p2p_gains_cmd_t import p2p_gains_cmd_t


class P2PGainsCmdSubscriber(BaseDataSubscriber):
    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = p2p_gains_cmd_t.decode(data)
        cmd = P2PGainsCmdData()
        cmd.set_data(
            float(msg.timestamp),
            float(msg.kp_scale),
            float(msg.kd_scale),
            float(msg.fjc_scale),
            float(msg.friction_phi),
            bool(msg.do_friction_comp),
        )
        self.data_queue.put(cmd)
