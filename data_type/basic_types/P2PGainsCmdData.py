"""Viser → controller gain / friction commands for min-jerk P2P."""

from __future__ import annotations

from data_type.BaseData import BaseData


class P2PGainsCmdData(BaseData):
    def __init__(self, name: str = "p2p_gains_cmd"):
        super().__init__(name)
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        self.fjc_scale = 1.0
        self.friction_phi = 0.03
        self.do_friction_comp = True

    def set_data(
        self,
        t: float,
        kp_scale: float,
        kd_scale: float,
        fjc_scale: float,
        friction_phi: float,
        do_friction_comp: bool,
    ) -> None:
        self.timestamp = float(t)
        self.kp_scale = float(kp_scale)
        self.kd_scale = float(kd_scale)
        self.fjc_scale = float(fjc_scale)
        self.friction_phi = float(friction_phi)
        self.do_friction_comp = bool(do_friction_comp)

    def get_data(self):
        return (
            self.timestamp,
            self.kp_scale,
            self.kd_scale,
            self.fjc_scale,
            self.friction_phi,
            self.do_friction_comp,
        )
