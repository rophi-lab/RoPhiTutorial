import numpy as np
import trimesh
import fcl
from yourdfpy import URDF
from viser.extras import ViserUrdf
import viser.transforms as vtf

from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)
from communication.lcm.subscriber.data_subscriber.ColInfoSubscriber import (
    ColInfoSubscriber,
)
from communication.lcm.subscriber.data_subscriber.Pose3DSubscriber import (
    Pose3DSubscriber,
)
from communication.lcm.subscriber.data_subscriber.JointTrajSubscriber import (
    JointTrajSubscriber,
)
from communication.lcm.subscriber.data_subscriber.P2PStatusSubscriber import (
    P2PStatusSubscriber,
)
from data_type.basic_types.JointMeasData import JointMeasData
from data_type.basic_types.ColInfoData import ColInfoData
from data_type.basic_types.Pose3DData import Pose3DData
from data_type.basic_types.JointTrajData import JointTrajData
from data_type.basic_types.P2PStatusData import MODE_NAME, P2PStatusData
from data_type.basic_types.ImpStatusData import (
    MODE_JOINT as IMP_MODE_JOINT,
    MODE_NAME as IMP_MODE_NAME,
    MODE_TASK as IMP_MODE_TASK,
    ImpStatusData,
)
from lcm_type.ctrl.p2p_gains_cmd_t import p2p_gains_cmd_t
from lcm_type.ctrl.imp_gains_cmd_t import imp_gains_cmd_t
from communication.lcm.subscriber.data_subscriber.ImpStatusSubscriber import (
    ImpStatusSubscriber,
)
from communication.lcm.subscriber.data_subscriber.SE3PoseSubscriber import (
    SE3PoseSubscriber,
)
from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from utils.fcl.getter import get_fcl_geom
from data_type.basic_types.SE3PoseData import SE3PoseData
from data_type.basic_types.NamedVecListData import NamedVecListData
from visualizer.data_visualizer.ContactArrowVisualizer import ContactArrowVisualizer

# Translucent RGBA colors for the geom overlay.
_ROBOT_COL_COLOR = [80, 140, 255, 110]  # blue-ish
_WALL_COL_COLOR = [235, 140, 45, 90]  # orange-ish

# Distance-line colors (RGB).
_ENV_DIST_COLOR = (255, 40, 40)  # red: robot <-> environment
_SELF_DIST_COLOR = (255, 150, 40)  # orange: self-collision
_VEL_ARROW_COLOR = (80, 200, 255)  # cyan: desired tip velocity
_TRAJ_PATH_COLOR = (255, 210, 40)  # yellow: planned EE path
_GOAL_MARKER_COLOR = (180, 80, 255)  # purple: sampled goal EE

_DASH_LEN = 0.012  # m
_GAP_LEN = 0.008  # m
_VEL_ARROW_SCALE = 0.4  # s: arrow length = ||v|| * scale
_VEL_ARROW_MIN = 1e-3  # m/s: hide below this
_VEL_ARROW_SHAFT = 0.004
_VEL_ARROW_HEAD_R = 0.008
_VEL_ARROW_HEAD_L = 0.02


def _primitive_mesh(geom_type, size, color):
    if geom_type == "cylinder":
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(size[1]))
    elif geom_type == "sphere":
        mesh = trimesh.creation.icosphere(radius=float(size[0]))
    elif geom_type == "box":
        mesh = trimesh.creation.box(extents=np.asarray(size, dtype=float))
    else:
        raise ValueError(f"Unsupported collision geom type: {geom_type}")
    mesh.visual.face_colors = color
    return mesh


def _geom_signature(dict_col_geoms):
    """Hashable summary of a col-geom set so we only rebuild meshes when the set
    (names/types/sizes) actually changes, not every message."""
    return tuple(
        (name, g["type"], tuple(np.round(np.asarray(g["size"], dtype=float), 5)))
        for name, g in sorted(dict_col_geoms.items())
    )


def _dashed_segments(p0, p1, dash_len=_DASH_LEN, gap_len=_GAP_LEN):
    """Return (N, 2, 3) dashed segments along the segment p0 -> p1."""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return np.zeros((0, 2, 3), dtype=np.float32)
    direction = v / length
    segs = []
    t = 0.0
    while t < length:
        t1 = min(t + dash_len, length)
        segs.append([p0 + direction * t, p0 + direction * t1])
        t = t1 + gap_len
    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segs, dtype=np.float32)


def _fcl_cache(dict_col_geoms):
    """Build name -> {fcl_geom, offset} from a ColInfo dict."""
    out = {}
    for name, g in dict_col_geoms.items():
        out[name] = {
            "fcl_geom": get_fcl_geom(**g),
            "offset": np.asarray(g["offset"], dtype=float).reshape(4, 4),
        }
    return out


class FlexivArmHandVisualizer:
    """Renders the Flexiv Rizon arm + Robotis RH-5 hand in viser, driven by two
    joint_meas_t LCM streams (arm and hand) from the sim env (or the hardware
    bridge + hand process).

    A "Show collision geometry" checkbox (off by default) overlays the collision
    geometry EXACTLY as it comes off the wire on the ``robot_col_info`` and
    ``static_col_info`` LCM channels - i.e. precisely what the collision-aware
    controller receives and reasons about. The static walls carry world-frame
    offsets and are drawn as-is; the robot primitives carry link-local offsets,
    so they are placed at ``FK(link) @ offset`` using the same joint stream the
    controller uses. If those channels aren't being published (e.g. running a
    non-collision env), the overlay is simply empty.

    A "Show collision distances" checkbox draws dashed nearest-point lines for
    configured collision pairs whose FCL distance is below ``col_dist_thr``:
    red for robot <-> environment, orange for self-collision.
    """

    def __init__(
        self,
        viser_server,
        lcm_instance,
        arm_joint_meas_data_queue,
        hand_joint_meas_data_queue,
        arm_joint_meas_channel: str,
        hand_joint_meas_channel: str,
        urdf_path: str,
        mesh_dir: str = "",
        num_arm_joints: int = 7,
        num_hand_joints: int = 20,
        robot_col_info_channel: str = "",
        static_col_info_channel: str = "",
        col_pairs=None,
        self_col_pairs=None,
        col_dist_thr: float = 0.1,
        target_pose_channel: str = "",
        target_joint_channel: str = "",
        traj_channel: str = "",
        traj_frame: str = "palm",
        status_channel: str = "",
        gains_cmd_channel: str = "",
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        fjc_scale: float = 1.0,
        friction_phi: float = 0.03,
        do_friction_comp: bool = True,
        plot_history_s: float = 8.0,
        impedance_status_channel: str = "",
        impedance_gains_cmd_channel: str = "",
        impedance_frame_axes_length: float = 0.08,
        kq_scale: float = 1.0,
        dq_scale: float = 1.0,
        kt_trans_scale: float = 1.0,
        kt_rot_scale: float = 1.0,
        dt_trans_scale: float = 1.0,
        dt_rot_scale: float = 1.0,
        palm_pose_channel: str = "",
        palm_frame_axes_length: float = 0.08,
        object_pose_channel: str = "",
        object_mesh_path: str = "",
        grasp_candidates_channel: str = "",
        fingertip_viz_channel: str = "",
        ft_path_viz_channel: str = "",
        contact_force_arrows_channel: str = "",
        contact_normal_arrows_channel: str = "",
        contact_force_arrow_scale: float = 0.02,
        contact_normal_arrow_scale: float = 0.03,
        *args,
        **kwargs,
    ):
        self._viser_server = viser_server
        self._lcm_instance = lcm_instance
        self._arm_queue = arm_joint_meas_data_queue
        self._hand_queue = hand_joint_meas_data_queue

        self._num_arm_joints = num_arm_joints
        self._num_hand_joints = num_hand_joints
        self._num_joints = num_arm_joints + num_hand_joints

        self._q = np.zeros(self._num_joints)

        self._col_pairs = [tuple(p) for p in (col_pairs or [])]
        self._self_col_pairs = [tuple(p) for p in (self_col_pairs or [])]
        self._col_dist_thr = float(col_dist_thr)
        self._traj_frame = str(traj_frame)

        JointMeasSubscriber(
            self._lcm_instance, arm_joint_meas_data_queue, num_arm_joints
        ).subscribe(arm_joint_meas_channel)
        JointMeasSubscriber(
            self._lcm_instance, hand_joint_meas_data_queue, num_hand_joints
        ).subscribe(hand_joint_meas_channel)

        urdf_kwargs = dict(
            load_meshes=True,
            build_scene_graph=True,
            load_collision_meshes=False,
            build_collision_scene_graph=False,
        )
        if mesh_dir:
            urdf_kwargs["mesh_dir"] = mesh_dir
        urdf = URDF.load(urdf_path, **urdf_kwargs)
        self._urdf = urdf  # kept for FK of the robot collision links
        self._viser_urdf = ViserUrdf(
            self._viser_server,
            urdf_or_path=urdf,
            load_meshes=True,
            load_collision_meshes=False,
        )

        self._setup_collision_overlay(robot_col_info_channel, static_col_info_channel)
        self._setup_target_pose(target_pose_channel)
        self._setup_p2p_overlays(
            urdf_path, mesh_dir, urdf_kwargs, target_joint_channel, traj_channel
        )
        self._setup_p2p_tracking_gui(
            status_channel=status_channel,
            gains_cmd_channel=gains_cmd_channel,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            fjc_scale=fjc_scale,
            friction_phi=friction_phi,
            do_friction_comp=do_friction_comp,
            plot_history_s=plot_history_s,
        )
        self._setup_impedance_gui(
            status_channel=impedance_status_channel,
            gains_cmd_channel=impedance_gains_cmd_channel,
            axes_length=float(impedance_frame_axes_length),
            kq_scale=kq_scale,
            dq_scale=dq_scale,
            kt_trans_scale=kt_trans_scale,
            kt_rot_scale=kt_rot_scale,
            dt_trans_scale=dt_trans_scale,
            dt_rot_scale=dt_rot_scale,
        )
        self._setup_palm_pose_overlay(
            palm_pose_channel,
            axes_length=float(palm_frame_axes_length),
            urdf_path=urdf_path,
            mesh_dir=mesh_dir,
            urdf_kwargs=urdf_kwargs,
        )
        self._setup_object_mesh_overlay(object_pose_channel, object_mesh_path)
        self._setup_grasp_viz_overlay(
            grasp_candidates_channel,
            fingertip_viz_channel,
            ft_path_viz_channel,
        )
        self._setup_contact_arrow_overlays(
            contact_force_arrows_channel,
            contact_normal_arrows_channel,
            force_scale=float(contact_force_arrow_scale),
            normal_scale=float(contact_normal_arrow_scale),
        )

    def _setup_target_pose(self, target_pose_channel: str):
        import queue as _queue

        self._target_pose_queue = _queue.Queue()
        self._target_handle = None
        self._tip_handle = None
        self._target_vel_handle = None
        if not target_pose_channel:
            return

        Pose3DSubscriber(self._lcm_instance, self._target_pose_queue).subscribe(
            target_pose_channel
        )
        self._target_handle = self._viser_server.scene.add_icosphere(
            "/ik/target",
            radius=0.012,
            color=(80, 220, 120),
            position=(0.0, 0.0, 0.0),
            visible=False,
        )
        # Current task point (published tip); distinct from the green target.
        self._tip_handle = self._viser_server.scene.add_icosphere(
            "/ik/tip",
            radius=0.008,
            color=(255, 80, 80),
            position=(0.0, 0.0, 0.0),
            visible=False,
        )
        # Placeholder segment; updated each poll from tip → tip + scale * v*.
        self._target_vel_handle = self._viser_server.scene.add_arrows(
            "/ik/target_vel",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.asarray(_VEL_ARROW_COLOR, dtype=np.uint8),
            shaft_radius=_VEL_ARROW_SHAFT,
            head_radius=_VEL_ARROW_HEAD_R,
            head_length=_VEL_ARROW_HEAD_L,
            visible=False,
        )

    def _setup_p2p_overlays(
        self, urdf_path, mesh_dir, urdf_kwargs, target_joint_channel, traj_channel
    ):
        """Ghost goal robot + planned EE path for min-jerk P2P."""
        import queue as _queue

        self._target_joint_queue = _queue.Queue()
        self._traj_queue = _queue.Queue()
        self._goal_urdf = None
        self._goal_marker = None
        self._traj_path_handle = None
        self._q_goal = None
        self._show_goal_robot = False

        if target_joint_channel:
            JointMeasSubscriber(
                self._lcm_instance, self._target_joint_queue, self._num_joints
            ).subscribe(target_joint_channel)
            goal_urdf = URDF.load(urdf_path, **urdf_kwargs)
            self._goal_urdf = ViserUrdf(
                self._viser_server,
                urdf_or_path=goal_urdf,
                root_node_name="/p2p/goal_robot",
                load_meshes=True,
                load_collision_meshes=False,
                # Translucent purple ghost of the sampled target config.
                mesh_color_override=(0.65, 0.35, 1.0, 0.35),
            )
            self._goal_urdf.show_visual = False

            self._goal_marker = self._viser_server.scene.add_icosphere(
                "/p2p/goal_ee",
                radius=0.015,
                color=_GOAL_MARKER_COLOR,
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            self._goal_robot_checkbox = self._viser_server.gui.add_checkbox(
                "Show goal robot", initial_value=True
            )

            @self._goal_robot_checkbox.on_update
            def _(_event):
                self._show_goal_robot = bool(self._goal_robot_checkbox.value)
                self._set_goal_robot_visible(
                    self._show_goal_robot
                    and self._q_goal is not None
                    and not getattr(self, "_float_hand_active", False)
                )

            self._show_goal_robot = True

        if traj_channel:
            JointTrajSubscriber(self._lcm_instance, self._traj_queue).subscribe(
                traj_channel
            )
            self._traj_path_handle = self._viser_server.scene.add_line_segments(
                "/p2p/traj_path",
                points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=_TRAJ_PATH_COLOR,
                line_width=3.0,
                visible=False,
            )

    def _setup_palm_pose_overlay(
        self,
        palm_pose_channel: str,
        axes_length: float,
        urdf_path: str = "",
        mesh_dir: str = "",
        urdf_kwargs: dict | None = None,
    ):
        """SE(3) frame + floating palm/finger ghost for IK-reach key ``1``."""
        import queue as _queue

        self._palm_pose_queue = _queue.Queue()
        self._palm_des_frame = None
        self._T_des = None
        self._float_hand_active = False
        self._q_float = None
        self._float_hand_base = None
        self._float_hand_viser = None
        self._float_hand_fk = None
        if not palm_pose_channel:
            return
        SE3PoseSubscriber(self._lcm_instance, self._palm_pose_queue).subscribe(
            palm_pose_channel
        )
        self._palm_des_frame = self._viser_server.scene.add_frame(
            "/ik_reach/T_des",
            axes_length=axes_length,
            axes_radius=0.004,
            origin_radius=0.01,
            origin_color=_GOAL_MARKER_COLOR,
            visible=False,
        )

        if not urdf_path:
            return
        kwargs = dict(urdf_kwargs or {})
        if mesh_dir:
            kwargs["mesh_dir"] = mesh_dir
        # Parent frame moved so FK(palm) lands on T_des; arm meshes hidden.
        self._float_hand_base = self._viser_server.scene.add_frame(
            "/ik_reach/float_hand",
            show_axes=False,
            visible=False,
        )
        float_urdf = URDF.load(urdf_path, **kwargs)
        self._float_hand_fk = float_urdf
        self._float_hand_viser = ViserUrdf(
            self._viser_server,
            urdf_or_path=float_urdf,
            root_node_name="/ik_reach/float_hand",
            load_meshes=True,
            load_collision_meshes=False,
            mesh_color_override=(0.65, 0.35, 1.0, 0.45),
        )
        self._float_hand_viser.show_visual = False
        self._apply_float_hand_mesh_visibility()

    @staticmethod
    def _is_float_hand_mesh(name: str) -> bool:
        """True for palm + finger meshes (scene path contains the palm link)."""
        return "palm" in str(name).split("/")

    def _apply_float_hand_mesh_visibility(self) -> None:
        """Hide arm/enclosure meshes; keep translucent palm + fingers only."""
        if self._float_hand_viser is None:
            return
        for mesh in self._float_hand_viser._meshes:
            mesh.visible = self._is_float_hand_mesh(mesh.name)

    def _set_float_hand_visible(self, visible: bool) -> None:
        if self._float_hand_base is None or self._float_hand_viser is None:
            return
        self._float_hand_base.visible = bool(visible)
        self._float_hand_viser.show_visual = bool(visible)
        if visible:
            # ``show_visual`` restores the visual root; re-hide non-hand meshes.
            self._apply_float_hand_mesh_visibility()

    def _update_float_hand_pose(self) -> None:
        """Place floating palm+fingers so the palm frame equals ``_T_des``."""
        if (
            not self._float_hand_active
            or self._T_des is None
            or self._q_float is None
            or self._float_hand_base is None
            or self._float_hand_viser is None
            or self._float_hand_fk is None
        ):
            self._set_float_hand_visible(False)
            return
        q = self._q_float
        self._float_hand_fk.update_cfg(q)
        T_palm = np.asarray(self._float_hand_fk.get_transform("palm"), dtype=float)
        T_base = self._T_des @ np.linalg.inv(T_palm)
        self._float_hand_viser.update_cfg(q)
        self._float_hand_base.position = tuple(T_base[:3, 3])
        self._float_hand_base.wxyz = tuple(vtf.SO3.from_matrix(T_base[:3, :3]).wxyz)
        self._set_float_hand_visible(True)

    def _poll_palm_pose(self) -> None:
        if not hasattr(self, "_palm_pose_queue") or self._palm_des_frame is None:
            return
        data = None
        while not self._palm_pose_queue.empty():
            data = self._palm_pose_queue.get()
        if not isinstance(data, SE3PoseData):
            return
        self._palm_des_frame.position = tuple(data.position)
        self._palm_des_frame.wxyz = tuple(data.quat_wxyz)
        self._palm_des_frame.visible = True
        # Rebuild T_des from published pose (matches controller SE(3)).
        R = vtf.SO3(wxyz=np.asarray(data.quat_wxyz, dtype=float)).as_matrix()
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(data.position, dtype=float)
        self._T_des = T
        if self._goal_marker is not None and self._float_hand_active:
            self._goal_marker.position = tuple(data.position)
            self._goal_marker.visible = True
        self._update_float_hand_pose()

    def _setup_object_mesh_overlay(
        self, object_pose_channel: str, object_mesh_path: str
    ) -> None:
        """Subscribe object SE(3) and render its mesh in Viser."""
        import queue as _queue

        self._object_pose_queue = _queue.Queue()
        self._object_mesh_handle = None
        self._object_frame = None
        if not object_pose_channel or not object_mesh_path:
            return
        SE3PoseSubscriber(self._lcm_instance, self._object_pose_queue).subscribe(
            object_pose_channel
        )
        mesh = trimesh.load(object_mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        self._object_frame = self._viser_server.scene.add_frame(
            "/grasp/object",
            show_axes=True,
            axes_length=0.06,
            axes_radius=0.003,
            visible=False,
        )
        self._object_mesh_handle = self._viser_server.scene.add_mesh_trimesh(
            "/grasp/object/mesh",
            mesh,
        )

    def _poll_object_pose(self) -> None:
        if (
            not hasattr(self, "_object_pose_queue")
            or self._object_frame is None
            or self._object_mesh_handle is None
        ):
            return
        data = None
        while not self._object_pose_queue.empty():
            data = self._object_pose_queue.get()
        if not isinstance(data, SE3PoseData):
            return
        self._object_frame.position = tuple(data.position)
        self._object_frame.wxyz = tuple(data.quat_wxyz)
        self._object_frame.visible = True

    def _setup_grasp_viz_overlay(
        self,
        grasp_candidates_channel: str,
        fingertip_viz_channel: str,
        ft_path_viz_channel: str = "",
    ) -> None:
        """Candidate antipodal pairs + fingertip cur/tgt + heuristic LVF paths."""
        import queue as _queue

        self._grasp_cand_queue = _queue.Queue()
        self._ft_viz_queue = _queue.Queue()
        self._ft_path_queue = _queue.Queue()
        self._grasp_pair_segs = None
        self._grasp_a_spheres: list = []
        self._grasp_b_spheres: list = []
        self._ft_cur_spheres: list = []
        self._ft_tgt_spheres: list = []
        self._ft_link_segs = None
        self._ft_path_segs: list = []

        if grasp_candidates_channel:
            NamedVecListSubscriber(
                self._lcm_instance, self._grasp_cand_queue
            ).subscribe(grasp_candidates_channel)
            self._grasp_pair_segs = self._viser_server.scene.add_line_segments(
                "/grasp/candidates/pairs",
                points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=(180, 180, 40),
                line_width=2.0,
                visible=False,
            )

        if fingertip_viz_channel:
            NamedVecListSubscriber(
                self._lcm_instance, self._ft_viz_queue
            ).subscribe(fingertip_viz_channel)
            colors_cur = [(255, 80, 80), (80, 180, 255), (80, 255, 120)]
            colors_tgt = [(255, 160, 160), (160, 210, 255), (160, 255, 180)]
            for i, name in enumerate(("thumb", "index", "middle")):
                self._ft_cur_spheres.append(
                    self._viser_server.scene.add_icosphere(
                        f"/grasp/ft/{name}_cur",
                        radius=0.008,
                        color=colors_cur[i],
                        position=(0.0, 0.0, 0.0),
                        visible=False,
                    )
                )
                self._ft_tgt_spheres.append(
                    self._viser_server.scene.add_icosphere(
                        f"/grasp/ft/{name}_tgt",
                        radius=0.01,
                        color=colors_tgt[i],
                        position=(0.0, 0.0, 0.0),
                        visible=False,
                    )
                )
            self._ft_link_segs = self._viser_server.scene.add_line_segments(
                "/grasp/ft/cur_to_tgt",
                points=np.zeros((3, 2, 3), dtype=np.float32),
                colors=(200, 200, 200),
                line_width=1.5,
                visible=False,
            )

        if ft_path_viz_channel:
            NamedVecListSubscriber(
                self._lcm_instance, self._ft_path_queue
            ).subscribe(ft_path_viz_channel)
            # Heuristic LVF polylines (distinct from straight cur→tgt).
            path_colors = [(255, 60, 60), (60, 140, 255), (40, 220, 100)]
            for i, name in enumerate(("thumb", "index", "middle")):
                self._ft_path_segs.append(
                    self._viser_server.scene.add_line_segments(
                        f"/grasp/ft_paths/{name}",
                        points=np.zeros((1, 2, 3), dtype=np.float32),
                        colors=path_colors[i],
                        line_width=4.0,
                        visible=False,
                    )
                )

    def _ensure_grasp_pair_spheres(self, n_pairs: int) -> None:
        while len(self._grasp_a_spheres) < n_pairs:
            i = len(self._grasp_a_spheres)
            self._grasp_a_spheres.append(
                self._viser_server.scene.add_icosphere(
                    f"/grasp/candidates/a_{i}",
                    radius=0.006,
                    color=(220, 180, 40),
                    position=(0.0, 0.0, 0.0),
                    visible=False,
                )
            )
            self._grasp_b_spheres.append(
                self._viser_server.scene.add_icosphere(
                    f"/grasp/candidates/b_{i}",
                    radius=0.006,
                    color=(40, 180, 220),
                    position=(0.0, 0.0, 0.0),
                    visible=False,
                )
            )

    def _poll_grasp_candidates(self) -> None:
        if not hasattr(self, "_grasp_cand_queue") or self._grasp_pair_segs is None:
            return
        data = None
        while not self._grasp_cand_queue.empty():
            data = self._grasp_cand_queue.get()
        if not isinstance(data, NamedVecListData):
            return
        _, names, vecs = data.get_data()
        # Expect c{i}_a / c{i}_b alternating.
        pts = np.asarray(vecs, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] < 2 or pts.shape[0] % 2 != 0:
            return
        n_pairs = pts.shape[0] // 2
        self._ensure_grasp_pair_spheres(n_pairs)
        segs = np.zeros((n_pairs, 2, 3), dtype=np.float32)
        for i in range(n_pairs):
            a = pts[2 * i]
            b = pts[2 * i + 1]
            segs[i, 0] = a
            segs[i, 1] = b
            self._grasp_a_spheres[i].position = tuple(a)
            self._grasp_b_spheres[i].position = tuple(b)
            self._grasp_a_spheres[i].visible = True
            self._grasp_b_spheres[i].visible = True
        for i in range(n_pairs, len(self._grasp_a_spheres)):
            self._grasp_a_spheres[i].visible = False
            self._grasp_b_spheres[i].visible = False
        self._grasp_pair_segs.points = segs
        self._grasp_pair_segs.visible = True

    def _poll_fingertip_viz(self) -> None:
        if not hasattr(self, "_ft_viz_queue") or not self._ft_cur_spheres:
            return
        data = None
        while not self._ft_viz_queue.empty():
            data = self._ft_viz_queue.get()
        if not isinstance(data, NamedVecListData):
            return
        _, names, vecs = data.get_data()
        name_to_p = {
            str(n): np.asarray(v, dtype=float).reshape(3)
            for n, v in zip(names, vecs)
        }
        keys_cur = ("thumb_cur", "index_cur", "middle_cur")
        keys_tgt = ("thumb_tgt", "index_tgt", "middle_tgt")
        segs = np.zeros((3, 2, 3), dtype=np.float32)
        for i, (kc, kt) in enumerate(zip(keys_cur, keys_tgt)):
            if kc not in name_to_p or kt not in name_to_p:
                continue
            pc, pt = name_to_p[kc], name_to_p[kt]
            self._ft_cur_spheres[i].position = tuple(pc)
            self._ft_tgt_spheres[i].position = tuple(pt)
            self._ft_cur_spheres[i].visible = True
            self._ft_tgt_spheres[i].visible = True
            segs[i, 0] = pc
            segs[i, 1] = pt
        if self._ft_link_segs is not None:
            self._ft_link_segs.points = segs
            self._ft_link_segs.visible = True

    def _poll_ft_paths(self) -> None:
        """Draw heuristic LVF fingertip polylines from NamedVecList waypoints."""
        if not getattr(self, "_ft_path_segs", None):
            return
        data = None
        while not self._ft_path_queue.empty():
            data = self._ft_path_queue.get()
        if not isinstance(data, NamedVecListData):
            return
        _, names, vecs = data.get_data()
        pts = np.asarray(vecs, dtype=np.float32).reshape(-1, 3)
        by_finger = {"thumb": [], "index": [], "middle": []}
        indexed = []
        for name, p in zip(names, pts):
            parts = str(name).split("_", 1)
            if len(parts) != 2 or parts[0] not in by_finger:
                continue
            try:
                ti = int(parts[1])
            except ValueError:
                continue
            indexed.append((parts[0], ti, p))
        indexed.sort(key=lambda t: (t[0], t[1]))
        for fname, _ti, p in indexed:
            by_finger[fname].append(p)

        for i, fname in enumerate(("thumb", "index", "middle")):
            path = by_finger[fname]
            handle = self._ft_path_segs[i]
            if len(path) < 2:
                handle.visible = False
                continue
            segs = np.zeros((len(path) - 1, 2, 3), dtype=np.float32)
            for j in range(len(path) - 1):
                segs[j, 0] = path[j]
                segs[j, 1] = path[j + 1]
            handle.points = segs
            handle.visible = True

    def _setup_contact_arrow_overlays(
        self,
        contact_force_arrows_channel: str,
        contact_normal_arrows_channel: str,
        force_scale: float,
        normal_scale: float,
    ) -> None:
        """Live contact force (magenta) and normal (cyan) arrows from controller."""
        import queue as _queue

        self._contact_arrow_vizs: list = []
        if contact_force_arrows_channel:
            q = _queue.Queue()
            self._contact_arrow_vizs.append(
                ContactArrowVisualizer(
                    self._viser_server,
                    self._lcm_instance,
                    q,
                    contact_force_arrows_channel,
                    scene_name="contact_forces",
                    scale=force_scale,
                    color=(1.0, 0.0, 0.85),
                    line_width=3.0,
                    point_radius=0.004,
                    normalize_direction=False,
                )
            )
        if contact_normal_arrows_channel:
            q = _queue.Queue()
            self._contact_arrow_vizs.append(
                ContactArrowVisualizer(
                    self._viser_server,
                    self._lcm_instance,
                    q,
                    contact_normal_arrows_channel,
                    scene_name="contact_normals",
                    scale=normal_scale,
                    color=(0.1, 0.85, 0.95),
                    line_width=2.0,
                    point_radius=0.003,
                    normalize_direction=True,
                )
            )

    def _poll_contact_arrows(self) -> None:
        for viz in getattr(self, "_contact_arrow_vizs", []):
            viz.update()

    def _setup_p2p_tracking_gui(
        self,
        status_channel: str,
        gains_cmd_channel: str,
        kp_scale: float,
        kd_scale: float,
        fjc_scale: float,
        friction_phi: float,
        do_friction_comp: bool,
        plot_history_s: float,
    ):
        """Arm q/q_des plots + gain/friction sliders (LCM ↔ controller)."""
        import queue as _queue

        import viser.uplot as uplot

        self._status_queue = _queue.Queue()
        self._gains_cmd_channel = str(gains_cmd_channel or "")
        self._plot_joint_idx = 0
        self._plot_buf_len = max(64, int(float(plot_history_s) * 30))
        self._plot_t = np.full(self._plot_buf_len, np.nan, dtype=np.float64)
        self._plot_q = np.full(self._plot_buf_len, np.nan, dtype=np.float64)
        self._plot_qd = np.full(self._plot_buf_len, np.nan, dtype=np.float64)
        self._plot_i = 0
        self._plot_fill = 0
        self._last_status = None
        self._status_md = None
        self._track_plot = None
        self._suppress_gains_pub = False

        if not status_channel:
            return

        P2PStatusSubscriber(self._lcm_instance, self._status_queue).subscribe(
            status_channel
        )

        with self._viser_server.gui.add_folder("P2P Tracking"):
            self._status_md = self._viser_server.gui.add_markdown(
                "_waiting for controller status…_"
            )
            self._plot_joint_dropdown = self._viser_server.gui.add_dropdown(
                "Arm joint plot",
                options=[f"joint{i}" for i in range(1, self._num_arm_joints + 1)],
                initial_value="joint1",
            )

            @self._plot_joint_dropdown.on_update
            def _(_event):
                label = str(self._plot_joint_dropdown.value)
                try:
                    self._plot_joint_idx = int(label.replace("joint", "")) - 1
                except ValueError:
                    self._plot_joint_idx = 0
                self._reset_track_plot_buffer()

            t0 = np.zeros(2, dtype=np.float64)
            y0 = np.zeros(2, dtype=np.float64)
            self._track_plot = self._viser_server.gui.add_uplot(
                data=(t0, y0, y0.copy()),
                series=(
                    uplot.Series(label="t"),
                    uplot.Series(label="q", stroke="#2aa198", width=2),
                    uplot.Series(label="q_des", stroke="#cb4b16", width=2),
                ),
                title="Arm joint tracking",
                aspect=2.2,
                height=220,
            )

        with self._viser_server.gui.add_folder("P2P Gains"):
            self._kp_scale_slider = self._viser_server.gui.add_slider(
                "kp_scale",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=float(kp_scale),
            )
            self._kd_scale_slider = self._viser_server.gui.add_slider(
                "kd_scale",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=float(kd_scale),
            )
            self._fjc_scale_slider = self._viser_server.gui.add_slider(
                "fjc_scale",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=float(fjc_scale),
            )
            self._phi_slider = self._viser_server.gui.add_slider(
                "friction_phi",
                min=0.005,
                max=0.2,
                step=0.005,
                initial_value=float(friction_phi),
            )
            self._fric_checkbox = self._viser_server.gui.add_checkbox(
                "do_friction_comp", initial_value=bool(do_friction_comp)
            )

            def _on_gain(_event=None):
                if self._suppress_gains_pub:
                    return
                self._publish_gains_cmd()

            self._kp_scale_slider.on_update(_on_gain)
            self._kd_scale_slider.on_update(_on_gain)
            self._fjc_scale_slider.on_update(_on_gain)
            self._phi_slider.on_update(_on_gain)
            self._fric_checkbox.on_update(_on_gain)

        # Push initial gains so controller / GUI stay aligned.
        self._publish_gains_cmd()

    def _reset_track_plot_buffer(self) -> None:
        self._plot_t[:] = np.nan
        self._plot_q[:] = np.nan
        self._plot_qd[:] = np.nan
        self._plot_i = 0
        self._plot_fill = 0

    def _publish_gains_cmd(self) -> None:
        if not self._gains_cmd_channel:
            return
        if not hasattr(self, "_kp_scale_slider"):
            return
        import time as _time

        msg = p2p_gains_cmd_t()
        msg.timestamp = float(_time.time())
        msg.kp_scale = float(self._kp_scale_slider.value)
        msg.kd_scale = float(self._kd_scale_slider.value)
        msg.fjc_scale = float(self._fjc_scale_slider.value)
        msg.friction_phi = float(self._phi_slider.value)
        msg.do_friction_comp = bool(self._fric_checkbox.value)
        self._lcm_instance.publish(self._gains_cmd_channel, msg.encode())

    def _poll_p2p_status(self) -> None:
        if not hasattr(self, "_status_queue"):
            return
        data = None
        while not self._status_queue.empty():
            data = self._status_queue.get()
        if not isinstance(data, P2PStatusData):
            return
        self._last_status = data
        self._update_status_markdown(data)
        self._append_track_plot_sample(data)

    def _update_status_markdown(self, data: P2PStatusData) -> None:
        if self._status_md is None:
            return
        n_arm = max(1, int(data.num_arm))
        mode = MODE_NAME.get(int(data.mode), str(data.mode))
        kp_arm = data.kp[:n_arm]
        kd_arm = data.kd[:n_arm]
        fjc_arm = data.fjc[:n_arm]
        err = float(np.linalg.norm(data.q[:n_arm] - data.q_des[:n_arm]))
        self._status_md.content = (
            f"**mode:** `{mode}`  \n"
            f"**kp_scale / kd_scale / fjc_scale:** "
            f"`{data.kp_scale:.2f}` / `{data.kd_scale:.2f}` / `{data.fjc_scale:.2f}`  \n"
            f"**arm kp (eff):** `[{', '.join(f'{v:.1f}' for v in kp_arm)}]`  \n"
            f"**arm kd (eff):** `[{', '.join(f'{v:.2f}' for v in kd_arm)}]`  \n"
            f"**arm Fjc (eff):** `[{', '.join(f'{v:.2f}' for v in fjc_arm)}]`  \n"
            f"**friction_phi:** `{data.friction_phi:.3f}`  "
            f"fric=`{data.do_friction_comp}`  grav=`{data.do_grav_comp}`  \n"
            f"**||q−q_des||_arm:** `{err:.4f}` rad"
        )

    def _append_track_plot_sample(self, data: P2PStatusData) -> None:
        if self._track_plot is None:
            return
        j = int(np.clip(self._plot_joint_idx, 0, self._num_arm_joints - 1))
        if j >= data.q.size:
            return
        i = self._plot_i % self._plot_buf_len
        self._plot_t[i] = float(data.timestamp)
        self._plot_q[i] = float(data.q[j])
        self._plot_qd[i] = float(data.q_des[j])
        self._plot_i += 1
        self._plot_fill = min(self._plot_buf_len, self._plot_fill + 1)
        if self._plot_fill < 2:
            return
        # Reconstruct chronological order from ring buffer.
        if self._plot_fill < self._plot_buf_len:
            t = self._plot_t[: self._plot_fill]
            q = self._plot_q[: self._plot_fill]
            qd = self._plot_qd[: self._plot_fill]
        else:
            order = (np.arange(self._plot_buf_len) + self._plot_i) % self._plot_buf_len
            t = self._plot_t[order]
            q = self._plot_q[order]
            qd = self._plot_qd[order]
        t0 = float(t[0])
        self._track_plot.data = (
            np.asarray(t - t0, dtype=np.float64),
            np.asarray(q, dtype=np.float64),
            np.asarray(qd, dtype=np.float64),
        )
        self._track_plot.title = f"Arm joint{j + 1}: q vs q_des"

    def _setup_impedance_gui(
        self,
        status_channel: str,
        gains_cmd_channel: str,
        axes_length: float,
        kq_scale: float,
        dq_scale: float,
        kt_trans_scale: float,
        kt_rot_scale: float,
        dt_trans_scale: float,
        dt_rot_scale: float,
    ):
        """Nominal vs current SE(3) frames + gain / mode tuning for impedance."""
        import queue as _queue
        import time as _time

        self._imp_status_queue = _queue.Queue()
        self._imp_gains_cmd_channel = str(gains_cmd_channel or "")
        self._imp_status_md = None
        self._imp_frame_nom = None
        self._imp_frame_cur = None
        self._last_imp_status = None
        self._suppress_imp_gains_pub = False

        if not status_channel:
            return

        ImpStatusSubscriber(self._lcm_instance, self._imp_status_queue).subscribe(
            status_channel
        )

        self._imp_frame_nom = self._viser_server.scene.add_frame(
            "/imp/T_nom",
            axes_length=axes_length,
            axes_radius=0.004,
            origin_radius=0.008,
            origin_color=(180, 80, 255),
            visible=True,
        )
        self._imp_frame_cur = self._viser_server.scene.add_frame(
            "/imp/T",
            axes_length=axes_length * 0.85,
            axes_radius=0.0035,
            origin_radius=0.007,
            origin_color=(80, 200, 120),
            visible=True,
        )

        with self._viser_server.gui.add_folder("Impedance"):
            self._imp_status_md = self._viser_server.gui.add_markdown(
                "_waiting for impedance status…_"
            )
            self._imp_mode_dropdown = self._viser_server.gui.add_dropdown(
                "Mode",
                options=["joint", "task"],
                initial_value="joint",
            )

            @self._imp_mode_dropdown.on_update
            def _(_event):
                if self._suppress_imp_gains_pub:
                    return
                self._publish_imp_gains_cmd()

            self._kq_scale_slider = self._viser_server.gui.add_slider(
                "kq_scale (joint)", min=0.0, max=3.0, step=0.05, initial_value=float(kq_scale)
            )
            self._dq_scale_slider = self._viser_server.gui.add_slider(
                "dq_scale (joint)", min=0.0, max=3.0, step=0.05, initial_value=float(dq_scale)
            )
            self._kt_trans_slider = self._viser_server.gui.add_slider(
                "kt_trans_scale",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=float(kt_trans_scale),
            )
            self._kt_rot_slider = self._viser_server.gui.add_slider(
                "kt_rot_scale",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=float(kt_rot_scale),
            )
            self._dt_trans_slider = self._viser_server.gui.add_slider(
                "dt_trans_scale",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=float(dt_trans_scale),
            )
            self._dt_rot_slider = self._viser_server.gui.add_slider(
                "dt_rot_scale",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=float(dt_rot_scale),
            )

            def _on_gain(_event=None):
                if self._suppress_imp_gains_pub:
                    return
                self._publish_imp_gains_cmd()

            for sl in (
                self._kq_scale_slider,
                self._dq_scale_slider,
                self._kt_trans_slider,
                self._kt_rot_slider,
                self._dt_trans_slider,
                self._dt_rot_slider,
            ):
                sl.on_update(_on_gain)

        self._publish_imp_gains_cmd()
        self._imp_boot_t = _time.time()

    def _publish_imp_gains_cmd(self) -> None:
        if not self._imp_gains_cmd_channel:
            return
        if not hasattr(self, "_kq_scale_slider"):
            return
        import time as _time

        msg = imp_gains_cmd_t()
        msg.timestamp = float(_time.time())
        msg.kq_scale = float(self._kq_scale_slider.value)
        msg.dq_scale = float(self._dq_scale_slider.value)
        msg.kt_trans_scale = float(self._kt_trans_slider.value)
        msg.kt_rot_scale = float(self._kt_rot_slider.value)
        msg.dt_trans_scale = float(self._dt_trans_slider.value)
        msg.dt_rot_scale = float(self._dt_rot_slider.value)
        mode_name = str(self._imp_mode_dropdown.value).lower()
        msg.mode = IMP_MODE_TASK if mode_name.startswith("t") else IMP_MODE_JOINT
        self._lcm_instance.publish(self._imp_gains_cmd_channel, msg.encode())

    def _poll_imp_status(self) -> None:
        if not hasattr(self, "_imp_status_queue"):
            return
        data = None
        while not self._imp_status_queue.empty():
            data = self._imp_status_queue.get()
        if not isinstance(data, ImpStatusData):
            return
        self._last_imp_status = data

        if self._imp_frame_nom is not None:
            self._imp_frame_nom.position = tuple(data.p_nom)
            self._imp_frame_nom.wxyz = tuple(data.quat_nom_wxyz)
            self._imp_frame_nom.visible = True
        if self._imp_frame_cur is not None:
            self._imp_frame_cur.position = tuple(data.p)
            self._imp_frame_cur.wxyz = tuple(data.quat_wxyz)
            self._imp_frame_cur.visible = True

        if self._imp_status_md is not None:
            mode = IMP_MODE_NAME.get(int(data.mode), str(data.mode))
            n_arm = max(1, int(data.num_arm))
            e_q = float(np.linalg.norm(data.q[:n_arm] - data.q_nom[:n_arm]))
            e_xi = float(np.linalg.norm(data.xi))
            e_p = float(np.linalg.norm(data.p - data.p_nom))
            self._imp_status_md.content = (
                f"**mode:** `{mode}`  \n"
                f"**||q−q_nom||_arm:** `{e_q:.4f}` rad  \n"
                f"**||p−p_nom||:** `{e_p:.4f}` m  \n"
                f"**||ξ|| (SE3 log):** `{e_xi:.4f}`  \n"
                f"**ξ=(ν,ω):** `[{', '.join(f'{v:.3f}' for v in data.xi)}]`  \n"
                f"**kq/dq scale:** `{data.kq_scale:.2f}` / `{data.dq_scale:.2f}`  \n"
                f"**kt (trans/rot):** `{data.kt_trans:.1f}` / `{data.kt_rot:.2f}`  \n"
                f"**dt (trans/rot):** `{data.dt_trans:.1f}` / `{data.dt_rot:.2f}`"
            )

        # Keep dropdown in sync when mode changed from the controller keyboard.
        if hasattr(self, "_imp_mode_dropdown"):
            want = "task" if int(data.mode) == IMP_MODE_TASK else "joint"
            if str(self._imp_mode_dropdown.value) != want:
                self._suppress_imp_gains_pub = True
                try:
                    self._imp_mode_dropdown.value = want
                finally:
                    self._suppress_imp_gains_pub = False

    def _set_goal_robot_visible(self, visible: bool) -> None:
        if self._goal_urdf is None:
            return
        self._goal_urdf.show_visual = bool(visible)

    def _fk_frame_position(self, q: np.ndarray, frame: str) -> np.ndarray:
        self._urdf.update_cfg(q)
        T = self._urdf.get_transform(frame)
        return np.asarray(T[:3, 3], dtype=float).copy()

    def _poll_target_joint(self):
        if not hasattr(self, "_target_joint_queue"):
            return
        data = None
        while not self._target_joint_queue.empty():
            data = self._target_joint_queue.get()
        if not isinstance(data, JointMeasData):
            return
        _, q, _, tau = data.get_data()
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self._num_joints:
            return
        # tau[0] == 1 → floating palm+fingers preview (IK reach key 1).
        float_hand = bool(np.asarray(tau, dtype=float).reshape(-1)[0] > 0.5)
        self._q_goal = q
        if float_hand:
            self._float_hand_active = True
            self._q_float = q
            self._set_goal_robot_visible(False)
            self._update_float_hand_pose()
            if self._goal_marker is not None and self._T_des is not None:
                self._goal_marker.position = tuple(self._T_des[:3, 3])
                self._goal_marker.visible = True
            return

        self._float_hand_active = False
        self._set_float_hand_visible(False)
        if self._goal_urdf is not None:
            self._goal_urdf.update_cfg(q)
            self._set_goal_robot_visible(self._show_goal_robot)
        if self._goal_marker is not None:
            try:
                p = self._fk_frame_position(q, self._traj_frame)
            except Exception:
                p = self._fk_frame_position(q, "palm")
            self._goal_marker.position = tuple(p)
            self._goal_marker.visible = True

    def _poll_joint_traj(self):
        if self._traj_path_handle is None:
            return
        data = None
        while not self._traj_queue.empty():
            data = self._traj_queue.get()
        if not isinstance(data, JointTrajData):
            return
        _, _t, q_path, _qd = data.get_data()
        if q_path.shape[0] < 2:
            self._traj_path_handle.visible = False
            return
        pts = []
        frame = self._traj_frame
        for q in q_path:
            try:
                pts.append(self._fk_frame_position(q, frame))
            except Exception:
                pts.append(self._fk_frame_position(q, "palm"))
                frame = "palm"
        pts = np.asarray(pts, dtype=np.float32)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 3)
        self._traj_path_handle.points = segs
        self._traj_path_handle.colors = np.tile(
            np.asarray(_TRAJ_PATH_COLOR, dtype=np.uint8), (segs.shape[0], 2, 1)
        )
        self._traj_path_handle.visible = True

    def _poll_target_pose(self):
        if self._target_handle is None:
            return
        data = None
        while not self._target_pose_queue.empty():
            data = self._target_pose_queue.get()
        if not isinstance(data, Pose3DData):
            return
        _, p_des, velocity, tip = data.get_data()
        p_des = np.asarray(p_des, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        tip = np.asarray(tip, dtype=float).reshape(3)

        self._target_handle.position = tuple(p_des)
        self._target_handle.visible = True
        self._tip_handle.position = tuple(tip)
        self._tip_handle.visible = True

        if self._target_vel_handle is None:
            return
        speed = float(np.linalg.norm(velocity))
        if speed < _VEL_ARROW_MIN:
            self._target_vel_handle.visible = False
            return
        end = tip + velocity * _VEL_ARROW_SCALE
        self._target_vel_handle.points = np.asarray(
            [[tip, end]], dtype=np.float32
        )
        self._target_vel_handle.visible = True

    # ------------------------------------------------------------------
    # Collision overlay (subscribed off the wire)
    # ------------------------------------------------------------------
    def _setup_collision_overlay(self, robot_col_info_channel, static_col_info_channel):
        import queue as _queue

        # Subscribe to the SAME channels the controller consumes.
        self._robot_col_queue = _queue.Queue()
        self._static_col_queue = _queue.Queue()
        if robot_col_info_channel:
            ColInfoSubscriber(self._lcm_instance, self._robot_col_queue).subscribe(
                robot_col_info_channel
            )
        if static_col_info_channel:
            ColInfoSubscriber(self._lcm_instance, self._static_col_queue).subscribe(
                static_col_info_channel
            )

        # Live node / FCL state.
        self._robot_col_nodes = {}  # name -> (handle, link_name, offset)
        self._wall_col_nodes = {}  # name -> handle
        self._robot_col_sig = None
        self._static_col_sig = None
        self._fcl_robot = {}  # name -> {fcl_geom, offset}
        self._fcl_static = {}

        self._show_collision = False
        self._checkbox = self._viser_server.gui.add_checkbox(
            "Show collision geometry", initial_value=False
        )

        @self._checkbox.on_update
        def _(_event):
            self._show_collision = self._checkbox.value
            for handle, _ln, _off in self._robot_col_nodes.values():
                handle.visible = self._show_collision
            for handle in self._wall_col_nodes.values():
                handle.visible = self._show_collision
            if self._show_collision:
                self._update_robot_col_transforms()

        # Distance lines (batched into two handles).
        empty = np.zeros((1, 2, 3), dtype=np.float32)
        self._env_dist_handle = self._viser_server.scene.add_line_segments(
            "/collision/distances/env",
            points=empty,
            colors=_ENV_DIST_COLOR,
            line_width=2.5,
            visible=False,
        )
        self._self_dist_handle = self._viser_server.scene.add_line_segments(
            "/collision/distances/self",
            points=empty,
            colors=_SELF_DIST_COLOR,
            line_width=2.5,
            visible=False,
        )

        self._show_distances = False
        self._dist_checkbox = self._viser_server.gui.add_checkbox(
            "Show collision distances", initial_value=False
        )

        @self._dist_checkbox.on_update
        def _(_event):
            self._show_distances = self._dist_checkbox.value
            if not self._show_distances:
                self._env_dist_handle.visible = False
                self._self_dist_handle.visible = False
            else:
                self._update_collision_distances()

    def _drain_latest_col(self, q):
        data = None
        while not q.empty():
            data = q.get()
        return data if isinstance(data, ColInfoData) else None

    def _rebuild_robot_nodes(self, dict_col_geoms):
        for _h, _ln, _o in self._robot_col_nodes.values():
            _h.remove()
        self._robot_col_nodes = {}
        for name, g in dict_col_geoms.items():
            mesh = _primitive_mesh(g["type"], g["size"], _ROBOT_COL_COLOR)
            handle = self._viser_server.scene.add_mesh_trimesh(
                f"/collision/robot/{name}", mesh, visible=self._show_collision
            )
            # Robot geom name == pinocchio/URDF link frame; offset is link-local.
            self._robot_col_nodes[name] = (handle, name, np.asarray(g["offset"]))

    def _rebuild_wall_nodes(self, dict_col_geoms):
        for _h in self._wall_col_nodes.values():
            _h.remove()
        self._wall_col_nodes = {}
        for name, g in dict_col_geoms.items():
            mesh = _primitive_mesh(g["type"], g["size"], _WALL_COL_COLOR)
            T = np.asarray(g["offset"])  # world-frame SE3
            handle = self._viser_server.scene.add_mesh_trimesh(
                f"/collision/walls/{name}",
                mesh,
                wxyz=vtf.SO3.from_matrix(T[:3, :3]).wxyz,
                position=T[:3, 3],
                visible=self._show_collision,
            )
            self._wall_col_nodes[name] = handle

    def _poll_collision_info(self):
        robot = self._drain_latest_col(self._robot_col_queue)
        if robot is not None:
            _, geoms = robot.get_data()
            sig = _geom_signature(geoms)
            if sig != self._robot_col_sig:
                self._robot_col_sig = sig
                self._fcl_robot = _fcl_cache(geoms)
                self._rebuild_robot_nodes(geoms)

        static = self._drain_latest_col(self._static_col_queue)
        if static is not None:
            _, geoms = static.get_data()
            sig = _geom_signature(geoms)
            if sig != self._static_col_sig:
                self._static_col_sig = sig
                self._fcl_static = _fcl_cache(geoms)
                self._rebuild_wall_nodes(geoms)

    def _update_robot_col_transforms(self):
        """Place each robot collision primitive at FK(link) @ local_offset, using
        the same joint stream the controller uses."""
        if not self._robot_col_nodes:
            return
        self._urdf.update_cfg(self._q)
        for handle, link_name, offset in self._robot_col_nodes.values():
            T = self._urdf.get_transform(link_name) @ offset
            handle.position = T[:3, 3]
            handle.wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz

    def _robot_fcl_objects(self):
        """World-frame FCL CollisionObjects for every known robot primitive."""
        self._urdf.update_cfg(self._q)
        objs = {}
        for name, entry in self._fcl_robot.items():
            try:
                T_link = self._urdf.get_transform(name)
            except KeyError:
                continue
            T = T_link @ entry["offset"]
            objs[name] = fcl.CollisionObject(
                entry["fcl_geom"],
                fcl.Transform(T[:3, :3], T[:3, 3]),
            )
        return objs

    def _static_fcl_objects(self):
        objs = {}
        for name, entry in self._fcl_static.items():
            T = entry["offset"]
            objs[name] = fcl.CollisionObject(
                entry["fcl_geom"],
                fcl.Transform(T[:3, :3], T[:3, 3]),
            )
        return objs

    @staticmethod
    def _pair_nearest_points(obj_a, obj_b):
        req = fcl.DistanceRequest(enable_nearest_points=True)
        res = fcl.DistanceResult()
        fcl.distance(obj_a, obj_b, req, res)
        return float(res.min_distance), res.nearest_points[0], res.nearest_points[1]

    @staticmethod
    def _apply_dashed_handle(handle, segment_batches, color, visible):
        if not segment_batches:
            handle.visible = False
            return
        points = np.concatenate(segment_batches, axis=0)
        if points.shape[0] == 0:
            handle.visible = False
            return
        handle.points = points.astype(np.float32)
        handle.colors = np.tile(
            np.asarray(color, dtype=np.uint8), (points.shape[0], 2, 1)
        )
        handle.visible = visible

    def _update_collision_distances(self):
        """Draw dashed nearest-point lines for pairs under ``col_dist_thr``."""
        if not self._show_distances:
            return
        if not self._fcl_robot:
            self._env_dist_handle.visible = False
            self._self_dist_handle.visible = False
            return

        robot_objs = self._robot_fcl_objects()
        static_objs = self._static_fcl_objects()
        thr = self._col_dist_thr

        env_segs = []
        for a, b in self._col_pairs:
            if a not in robot_objs or b not in static_objs:
                continue
            dist, p0, p1 = self._pair_nearest_points(robot_objs[a], static_objs[b])
            if dist < thr:
                env_segs.append(_dashed_segments(p0, p1))

        self_segs = []
        for a, b in self._self_col_pairs:
            if a not in robot_objs or b not in robot_objs:
                continue
            dist, p0, p1 = self._pair_nearest_points(robot_objs[a], robot_objs[b])
            if dist < thr:
                self_segs.append(_dashed_segments(p0, p1))

        self._apply_dashed_handle(
            self._env_dist_handle, env_segs, _ENV_DIST_COLOR, True
        )
        self._apply_dashed_handle(
            self._self_dist_handle, self_segs, _SELF_DIST_COLOR, True
        )

    # ------------------------------------------------------------------
    def _drain_latest(self, queue):
        data = None
        while not queue.empty():
            data = queue.get()
        if data is not None and not isinstance(data, JointMeasData):
            raise ValueError(f"Expected JointMeasData on the queue, got {type(data)}")
        return data

    def _check_and_get_data_from_que(self):
        arm = self._drain_latest(self._arm_queue)
        if arm is not None:
            _, q, _, _ = arm.get_data()
            self._q[: self._num_arm_joints] = q

        hand = self._drain_latest(self._hand_queue)
        if hand is not None:
            _, q, _, _ = hand.get_data()
            self._q[self._num_arm_joints :] = q

    def update(self):
        self._check_and_get_data_from_que()
        self._viser_urdf.update_cfg(self._q)
        # Pick up any new collision geometry off the wire.
        self._poll_collision_info()
        self._poll_target_pose()
        self._poll_target_joint()
        self._poll_joint_traj()
        self._poll_p2p_status()
        self._poll_imp_status()
        self._poll_palm_pose()
        self._poll_object_pose()
        self._poll_grasp_candidates()
        self._poll_fingertip_viz()
        self._poll_ft_paths()
        self._poll_contact_arrows()
        # Only pay the FK cost for the overlay when it's actually shown.
        if self._show_collision:
            self._update_robot_col_transforms()
        if self._show_distances:
            self._update_collision_distances()
