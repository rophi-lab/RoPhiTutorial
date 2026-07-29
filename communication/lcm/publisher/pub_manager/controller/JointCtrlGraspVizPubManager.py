from queue import Queue

import lcm

from communication.lcm.publisher.BasePubManager import BaseCtrlPubManager
from communication.lcm.publisher.data_publisher.JointCtrlPublisher import (
    JointCtrlPublisher,
)
from communication.lcm.publisher.data_publisher.NamedVecListPublisher import (
    NamedVecListPublisher,
)


class JointCtrlGraspVizPubManager(BaseCtrlPubManager):
    """Joint ctrl + grasp / contact NamedVecList channels for Viser."""

    def __init__(
        self,
        ctrl_channel: str = "sw_flexiv_arm_hand_joint_ctrl",
        grasp_candidates_channel: str = "sw_grasp_candidates",
        fingertip_viz_channel: str = "sw_grasp_fingertips",
        ft_path_viz_channel: str = "",
        contact_force_arrows_channel: str = "",
        contact_normal_arrows_channel: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.publisher_dict[ctrl_channel] = JointCtrlPublisher(
            self._lcm_instance, ctrl_channel, self.ctrl_pub_que
        )
        for ch in (
            grasp_candidates_channel,
            fingertip_viz_channel,
            ft_path_viz_channel,
            contact_force_arrows_channel,
            contact_normal_arrows_channel,
        ):
            if not ch:
                continue
            self.pub_que_dict[ch] = Queue(maxsize=1)
            self.publisher_dict[ch] = NamedVecListPublisher(
                self._lcm_instance, ch, self.pub_que_dict[ch]
            )
