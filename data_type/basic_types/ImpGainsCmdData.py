"""Viser → controller gain / mode commands for impedance control."""

from __future__ import annotations

from data_type.BaseData import BaseData


class ImpGainsCmdData(BaseData):
    def __init__(self, name: str = "imp_gains_cmd"):
        super().__init__(name)
        self.kq_scale = 1.0
        self.dq_scale = 1.0
        self.kt_trans_scale = 1.0
        self.kt_rot_scale = 1.0
        self.dt_trans_scale = 1.0
        self.dt_rot_scale = 1.0
        self.mode = -1  # -1 = leave mode unchanged

    def set_data(
        self,
        t: float,
        kq_scale: float,
        dq_scale: float,
        kt_trans_scale: float,
        kt_rot_scale: float,
        dt_trans_scale: float,
        dt_rot_scale: float,
        mode: int = -1,
    ) -> None:
        self.timestamp = float(t)
        self.kq_scale = float(kq_scale)
        self.dq_scale = float(dq_scale)
        self.kt_trans_scale = float(kt_trans_scale)
        self.kt_rot_scale = float(kt_rot_scale)
        self.dt_trans_scale = float(dt_trans_scale)
        self.dt_rot_scale = float(dt_rot_scale)
        self.mode = int(mode)
