from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.P2PStatusData import P2PStatusData
from lcm_type.ctrl.p2p_status_t import p2p_status_t


class P2PStatusPublisher(BaseDataPublisher):
    """Publish min-jerk P2P status (q/q_des, gains, friction) for Viser."""

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, P2PStatusData):
            return
        msg = p2p_status_t()
        msg.timestamp = float(data.timestamp)
        msg.mode = int(data.mode)
        msg.num_joints = int(data.num_joints)
        msg.num_arm = int(data.num_arm)
        msg.q = data.q.astype(float).tolist()
        msg.q_des = data.q_des.astype(float).tolist()
        msg.qd = data.qd.astype(float).tolist()
        msg.qd_des = data.qd_des.astype(float).tolist()
        msg.kp = data.kp.astype(float).tolist()
        msg.kd = data.kd.astype(float).tolist()
        msg.fjc = data.fjc.astype(float).tolist()
        msg.friction_phi = float(data.friction_phi)
        msg.kp_scale = float(data.kp_scale)
        msg.kd_scale = float(data.kd_scale)
        msg.fjc_scale = float(data.fjc_scale)
        msg.do_friction_comp = bool(data.do_friction_comp)
        msg.do_grav_comp = bool(data.do_grav_comp)
        self._lcm_instance.publish(self._channel, msg.encode())
