"""Robot-free Viser manager: FoundationPose mesh estimate + optional GT ghost."""

from queue import Queue

from omegaconf import DictConfig

from visualizer.BaseVisManager import BaseVisManager
from visualizer.data_visualizer.MeshVisualizer import MeshVisualizer
from visualizer.data_visualizer.PoseVisualizer import PoseVisualizer


class ObjPoseVisManager(BaseVisManager):
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, max_sub_freq=1000, *args, **kwargs)

    def _set_visualizers(self, config: DictConfig):
        visualizers = []

        fp_cfg = config["foundation_pose_mesh"]
        fp_channel = fp_cfg["pose_data_channel"]
        # Separate queues so mesh + axes subscribers do not steal each other's msgs.
        fp_mesh_que = Queue()
        self.data_queue_dict["extr_sub_que_dict"][fp_channel] = fp_mesh_que
        visualizers.append(
            MeshVisualizer(
                self.viser,
                self._lcm_instance,
                pose_data_queue=fp_mesh_que,
                mesh_pose_data_channel=fp_channel,
                mesh_path=fp_cfg["mesh_path"],
                mesh_name=fp_cfg.get("mesh_name", "predefined_obj"),
                scene_name=fp_cfg.get("scene_name", "/fp_mesh"),
                opacity=float(fp_cfg.get("opacity", 1.0)),
            )
        )

        if bool(fp_cfg.get("show_axes", True)):
            fp_axes_que = Queue()
            self.data_queue_dict["extr_sub_que_dict"][fp_channel + "_axes"] = fp_axes_que
            visualizers.append(
                PoseVisualizer(
                    self.viser,
                    self._lcm_instance,
                    pose_data_queue=fp_axes_que,
                    pose_data_channel=fp_channel,
                    mesh_name=fp_cfg.get("mesh_name", "predefined_obj"),
                    scene_name=fp_cfg.get("axes_scene_name", "/fp_pose_frame"),
                    axes_length=float(fp_cfg.get("axes_length", 0.1)),
                )
            )

        gt_cfg = config.get("gt_mesh", None)
        if gt_cfg is not None:
            gt_channel = gt_cfg["pose_data_channel"]
            self.data_queue_dict["extr_sub_que_dict"][gt_channel] = Queue()
            visualizers.append(
                MeshVisualizer(
                    self.viser,
                    self._lcm_instance,
                    pose_data_queue=self.data_queue_dict["extr_sub_que_dict"][
                        gt_channel
                    ],
                    mesh_pose_data_channel=gt_channel,
                    mesh_path=gt_cfg["mesh_path"],
                    mesh_name=gt_cfg.get("mesh_name", "predefined_obj"),
                    scene_name=gt_cfg.get("scene_name", "/gt_mesh"),
                    opacity=float(gt_cfg.get("opacity", 0.35)),
                )
            )

        self._list_visualizers = visualizers
        print(
            f"[ObjPoseVisManager] Initialized with "
            f"sub freq: {self._max_sub_freq} Hz, vis freq: {self._vis_freq} Hz"
        )
