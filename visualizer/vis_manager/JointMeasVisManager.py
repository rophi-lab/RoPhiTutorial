from omegaconf import DictConfig
from queue import Queue

from visualizer.BaseVisManager import BaseVisManager
from visualizer.data_visualizer.JointMeasVisualizer import JointMeasVisualizer


class JointMeasVisManager(BaseVisManager):
    """
    Visualizer manager for live joint position and torque plots from a single LCM channel.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, max_sub_freq=1000, *args, **kwargs)

    def _set_visualizers(self, config: DictConfig):
        jm_cfg = config["joint_meas"]

        # Prepare a queue for the joint measurement channel
        self.data_queue_dict["intr_sub_que_dict"][
            jm_cfg["joint_meas_channel"]
        ] = Queue()

        jm_vis = JointMeasVisualizer(
            self.viser,
            self._lcm_instance,
            self.data_queue_dict["intr_sub_que_dict"][jm_cfg["joint_meas_channel"]],
            jm_cfg["joint_meas_channel"],
            num_joints=jm_cfg["num_joints"],
            joint_names=jm_cfg.get("joint_names", None),
            history_len=jm_cfg.get("history_len", 300),
            cols=jm_cfg.get("cols", 3),
            q_y_ranges=None,
            tau_y_ranges=None,
            tau_hpf_enable=jm_cfg.get("tau_hpf_enable", False),
            tau_hpf_cutoff_hz=jm_cfg.get("tau_hpf_cutoff_hz", 1.0),
            paired_joint_indices=jm_cfg.get("paired_joint_indices", [3, 4, 5]),
        )

        self._list_visualizers = [jm_vis]
        print(
            f"[Visualizer] JointMeasVisManager initialized with sub freq: {self._max_sub_freq} Hz, vis freq: {self._vis_freq} Hz."
        )
