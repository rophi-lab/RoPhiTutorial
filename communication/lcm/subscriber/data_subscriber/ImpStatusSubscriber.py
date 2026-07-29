from queue import Queue

import numpy as np
import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.ImpStatusData import ImpStatusData
from lcm_type.ctrl.imp_status_t import imp_status_t


class ImpStatusSubscriber(BaseDataSubscriber):
    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = imp_status_t.decode(data)
        n = int(msg.num_joints)
        status = ImpStatusData(num_joints=n)
        status.set_data(
            msg.timestamp,
            int(msg.mode),
            np.asarray(msg.q, dtype=np.float64),
            np.asarray(msg.q_nom, dtype=np.float64),
            np.asarray(msg.p, dtype=np.float64),
            np.asarray(msg.p_nom, dtype=np.float64),
            np.asarray(msg.quat_wxyz, dtype=np.float64),
            np.asarray(msg.quat_nom_wxyz, dtype=np.float64),
            np.asarray(msg.xi, dtype=np.float64),
            num_arm=int(msg.num_arm),
            kq_scale=float(msg.kq_scale),
            dq_scale=float(msg.dq_scale),
            kt_trans=float(msg.kt_trans),
            kt_rot=float(msg.kt_rot),
            dt_trans=float(msg.dt_trans),
            dt_rot=float(msg.dt_rot),
        )
        self.data_queue.put(status)
