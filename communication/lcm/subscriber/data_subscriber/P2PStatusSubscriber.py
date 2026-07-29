from queue import Queue

import numpy as np
import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.P2PStatusData import P2PStatusData
from lcm_type.ctrl.p2p_status_t import p2p_status_t


class P2PStatusSubscriber(BaseDataSubscriber):
    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = p2p_status_t.decode(data)
        n = int(msg.num_joints)
        status = P2PStatusData(num_joints=n)
        status.set_data(
            msg.timestamp,
            int(msg.mode),
            np.asarray(msg.q, dtype=np.float64),
            np.asarray(msg.q_des, dtype=np.float64),
            np.asarray(msg.qd, dtype=np.float64),
            np.asarray(msg.qd_des, dtype=np.float64),
            np.asarray(msg.kp, dtype=np.float64),
            np.asarray(msg.kd, dtype=np.float64),
            np.asarray(msg.fjc, dtype=np.float64),
            num_arm=int(msg.num_arm),
            friction_phi=float(msg.friction_phi),
            kp_scale=float(msg.kp_scale),
            kd_scale=float(msg.kd_scale),
            fjc_scale=float(msg.fjc_scale),
            do_friction_comp=bool(msg.do_friction_comp),
            do_grav_comp=bool(msg.do_grav_comp),
        )
        self.data_queue.put(status)
