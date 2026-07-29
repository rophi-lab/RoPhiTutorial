from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.ImpStatusData import ImpStatusData
from lcm_type.ctrl.imp_status_t import imp_status_t


class ImpStatusPublisher(BaseDataPublisher):
    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if self._data_que.empty():
            return
        data = self._data_que.get()
        if not isinstance(data, ImpStatusData):
            return
        msg = imp_status_t()
        msg.timestamp = float(data.timestamp)
        msg.mode = int(data.mode)
        msg.num_joints = int(data.num_joints)
        msg.num_arm = int(data.num_arm)
        msg.q = data.q.astype(float).tolist()
        msg.q_nom = data.q_nom.astype(float).tolist()
        msg.p = data.p.astype(float).tolist()
        msg.p_nom = data.p_nom.astype(float).tolist()
        msg.quat_wxyz = data.quat_wxyz.astype(float).tolist()
        msg.quat_nom_wxyz = data.quat_nom_wxyz.astype(float).tolist()
        msg.xi = data.xi.astype(float).tolist()
        msg.kq_scale = float(data.kq_scale)
        msg.dq_scale = float(data.dq_scale)
        msg.kt_trans = float(data.kt_trans)
        msg.kt_rot = float(data.kt_rot)
        msg.dt_trans = float(data.dt_trans)
        msg.dt_rot = float(data.dt_rot)
        self._lcm_instance.publish(self._channel, msg.encode())
