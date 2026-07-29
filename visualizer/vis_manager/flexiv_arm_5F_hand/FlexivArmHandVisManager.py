from queue import Queue

from omegaconf import DictConfig

from visualizer.BaseVisManager import BaseVisManager
from visualizer.data_visualizer.FlexivArmHandVisualizer import (
    FlexivArmHandVisualizer,
)


class FlexivArmHandVisManager(BaseVisManager):
    """Viser visualizer for the Flexiv Rizon arm + Robotis RH-5 hand.

    Subscribes to the arm (7) and hand (20) joint_meas_t channels the sim env
    publishes and renders the combined URDF in real time. The same channels are
    used on hardware (the bridge publishes the arm, the hand process the hand),
    so no config change is needed to switch sim <-> hardware.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, max_sub_freq=1000, *args, **kwargs)

    def _set_visualizers(self, config: DictConfig):
        fcfg = config["flexiv_arm_5F_hand"]
        arm_channel = fcfg["arm_joint_meas_channel"]
        hand_channel = fcfg["hand_joint_meas_channel"]

        intr = self.data_queue_dict["intr_sub_que_dict"]
        intr[arm_channel] = Queue()
        intr[hand_channel] = Queue()

        vis = FlexivArmHandVisualizer(
            self.viser,
            self._lcm_instance,
            intr[arm_channel],
            intr[hand_channel],
            arm_joint_meas_channel=arm_channel,
            hand_joint_meas_channel=hand_channel,
            urdf_path=fcfg["urdf_path"],
            mesh_dir=fcfg.get("mesh_dir", ""),
            num_arm_joints=fcfg.get("num_arm_joints", 7),
            num_hand_joints=fcfg.get("num_hand_joints", 20),
            robot_col_info_channel=fcfg.get("robot_col_info_channel", ""),
            static_col_info_channel=fcfg.get("static_col_info_channel", ""),
            col_pairs=fcfg.get("col_pairs", []),
            self_col_pairs=fcfg.get("self_col_pairs", []),
            col_dist_thr=float(fcfg.get("col_dist_thr", 0.1)),
            target_pose_channel=fcfg.get("target_pose_channel", ""),
            target_joint_channel=fcfg.get("target_joint_channel", ""),
            traj_channel=fcfg.get("traj_channel", ""),
            traj_frame=fcfg.get("traj_frame", "palm"),
            status_channel=fcfg.get("status_channel", ""),
            gains_cmd_channel=fcfg.get("gains_cmd_channel", ""),
            kp_scale=float(fcfg.get("kp_scale", 1.0)),
            kd_scale=float(fcfg.get("kd_scale", 1.0)),
            fjc_scale=float(fcfg.get("fjc_scale", 1.0)),
            friction_phi=float(fcfg.get("friction_phi", 0.03)),
            do_friction_comp=bool(fcfg.get("do_friction_comp", True)),
            plot_history_s=float(fcfg.get("plot_history_s", 8.0)),
            impedance_status_channel=fcfg.get("impedance_status_channel", ""),
            impedance_gains_cmd_channel=fcfg.get(
                "impedance_gains_cmd_channel", ""
            ),
            impedance_frame_axes_length=float(
                fcfg.get("impedance_frame_axes_length", 0.08)
            ),
            kq_scale=float(fcfg.get("kq_scale", 1.0)),
            dq_scale=float(fcfg.get("dq_scale", 1.0)),
            kt_trans_scale=float(fcfg.get("kt_trans_scale", 1.0)),
            kt_rot_scale=float(fcfg.get("kt_rot_scale", 1.0)),
            dt_trans_scale=float(fcfg.get("dt_trans_scale", 1.0)),
            dt_rot_scale=float(fcfg.get("dt_rot_scale", 1.0)),
            palm_pose_channel=fcfg.get("palm_pose_channel", ""),
            palm_frame_axes_length=float(
                fcfg.get("palm_frame_axes_length", 0.08)
            ),
            object_pose_channel=fcfg.get("object_pose_channel", ""),
            object_mesh_path=fcfg.get("object_mesh_path", ""),
            grasp_candidates_channel=fcfg.get("grasp_candidates_channel", ""),
            fingertip_viz_channel=fcfg.get("fingertip_viz_channel", ""),
            ft_path_viz_channel=fcfg.get("ft_path_viz_channel", ""),
            contact_force_arrows_channel=fcfg.get(
                "contact_force_arrows_channel", ""
            ),
            contact_normal_arrows_channel=fcfg.get(
                "contact_normal_arrows_channel", ""
            ),
            contact_force_arrow_scale=float(
                fcfg.get("contact_force_arrow_scale", 0.02)
            ),
            contact_normal_arrow_scale=float(
                fcfg.get("contact_normal_arrow_scale", 0.03)
            ),
        )

        self._list_visualizers = [vis]
        print(
            f"[Visualizer] FlexivArmHandVisManager initialized "
            f"(sub freq={self._max_sub_freq} Hz, vis freq={self._vis_freq} Hz)."
        )
