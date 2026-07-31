"""Visualizer for the MuJoCo eval grasp system (hand-only, no arm).

Uses RealTimeMeshGraspVisualizer which renders:
- Object mesh with live pose
- Hand URDF with live joint positions
- Per-contact force arrows (magenta)

Plus a separate ContactForceArrowVisualizer for the net force arrow (green).
"""

from visualizer.BaseVisManager import BaseVisManager
from omegaconf import DictConfig
from queue import Queue

from visualizer.data_visualizer.RealTimeMeshGraspVisualizer import (
    RealTimeMeshGraspVisualizer,
)
from visualizer.data_visualizer.ContactForceArrowVisualizer import (
    ContactForceArrowVisualizer,
)


class MjEvalGraspVisManager(BaseVisManager):

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, max_sub_freq=1000, *args, **kwargs)

    def _set_visualizers(self, config: DictConfig):
        rt_config = config["realtime_mesh_grasp"]

        # Object pose queue
        obj_pose_channel = rt_config["obj_pose_data_channel"]
        self.data_queue_dict["extr_sub_que_dict"][obj_pose_channel] = Queue()

        # Hand joint queue
        hand_joint_channel = rt_config["hand_joint_data_channel"]
        self.data_queue_dict["intr_sub_que_dict"][hand_joint_channel] = Queue()

        # Contact forces queue
        contact_forces_channel = rt_config.get("contact_forces_data_channel", None)
        contact_forces_queue = None
        if contact_forces_channel:
            self.data_queue_dict["extr_sub_que_dict"][contact_forces_channel] = Queue()
            contact_forces_queue = self.data_queue_dict["extr_sub_que_dict"][
                contact_forces_channel
            ]

        rt_visualizer = RealTimeMeshGraspVisualizer(
            self.viser,
            self._lcm_instance,
            obj_pose_data_queue=self.data_queue_dict["extr_sub_que_dict"][
                obj_pose_channel
            ],
            obj_pose_data_channel=obj_pose_channel,
            hand_joint_data_queue=self.data_queue_dict["intr_sub_que_dict"][
                hand_joint_channel
            ],
            hand_joint_data_channel=hand_joint_channel,
            num_hand_joints=rt_config.get("num_hand_joints", 20),
            contact_forces_data_queue=contact_forces_queue,
            contact_forces_data_channel=contact_forces_channel,
            mesh_path=rt_config["mesh_path"],
            obj_name=rt_config.get("obj_name", ""),
            mesh_name=rt_config.get("mesh_name", "predefined_obj"),
            hand_urdf_path=rt_config["hand_urdf_path"],
            hand_mesh_path=rt_config["hand_mesh_path"],
            hand_alpha=rt_config.get("hand_alpha", 0.5),
            force_scale=rt_config.get("force_scale", 0.05),
        )

        visualizers = [rt_visualizer]

        # Net force arrow (green, separate channel)
        net_force_config = config.get("net_force_arrow", None)
        if net_force_config:
            net_force_channel = net_force_config["contact_force_arrows_data_channel"]
            self.data_queue_dict["extr_sub_que_dict"][net_force_channel] = Queue()

            net_force_visualizer = ContactForceArrowVisualizer(
                self.viser,
                self._lcm_instance,
                contact_force_arrows_data_queue=self.data_queue_dict[
                    "extr_sub_que_dict"
                ][net_force_channel],
                contact_force_arrows_data_channel=net_force_channel,
                force_scale=net_force_config.get("force_scale", 0.2),
                color=tuple(net_force_config.get("color", [0.0, 1.0, 0.2])),
                line_width=net_force_config.get("line_width", 4.0),
            )
            visualizers.append(net_force_visualizer)

        self._list_visualizers = visualizers
        print(
            f"[MjEvalGraspVisManager] Initialized with "
            f"sub freq: {self._max_sub_freq} Hz, vis freq: {self._vis_freq} Hz"
        )
