import queue as Queue

from communication.lcm.subscriber.BaseSubManager import BaseEnvSubManeager
from communication.lcm.subscriber.data_subscriber.JointCtrlSubsriber import (
    JointCtrlSubscriber,
)


class JointSubManager(BaseEnvSubManeager):
    """
    This class manages the subscribers for the joints.
    It inherits from the BaseSubManager class.
    """

    def __init__(
        self, ctrl_channel: str = "ctrl_data", num_joints: int = 7, *args, **kwargs
    ):
        """
        Initialize the JointSubManager.
        @param[in] ctrl_channel: The channel name for the control data.
        """
        super().__init__(*args, **kwargs)
        self.num_joints = num_joints

        ctrl_sub = JointCtrlSubscriber(
            self._lcm_instance,
            self.data_queue_dict["ctrl_sub_que"],
            num_joints=self.num_joints,
        )
        ctrl_sub.subscribe(ctrl_channel)
