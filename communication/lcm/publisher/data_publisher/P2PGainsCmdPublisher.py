from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.P2PGainsCmdData import P2PGainsCmdData
from lcm_type.ctrl.p2p_gains_cmd_t import p2p_gains_cmd_t


class P2PGainsCmdPublisher(BaseDataPublisher):
    """Publish gain / friction commands from Viser to the controller."""

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, P2PGainsCmdData):
            return
        msg = p2p_gains_cmd_t()
        msg.timestamp = float(data.timestamp)
        msg.kp_scale = float(data.kp_scale)
        msg.kd_scale = float(data.kd_scale)
        msg.fjc_scale = float(data.fjc_scale)
        msg.friction_phi = float(data.friction_phi)
        msg.do_friction_comp = bool(data.do_friction_comp)
        self._lcm_instance.publish(self._channel, msg.encode())
