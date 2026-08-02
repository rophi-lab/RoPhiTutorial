"""No-op sim platform for camera / object-only scenes (no robot)."""

from __future__ import annotations

from robot_platform.sim.BaseSimPlatform import BaseSimPlatform


class EmptySimPlatform(BaseSimPlatform):
    """Platform stub when the MJCF scene has no actuated robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_valid = True

    def set_mj_data_name_idx(self, mj_data=None):
        """No joints to index."""

    def apply_sim_control(self, ctrl_data, mj_data):
        """No actuators."""

    def sync_intr_data_from_sim(self, mj_data, intr_pub_que_dict: dict):
        """No intrinsic robot measurements."""

    def reset(self, mj_data=None):
        """Nothing to reset on the platform side."""

    def get_col_info_data(self):
        return []
