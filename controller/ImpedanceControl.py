"""Joint-space and SE(3) task-space impedance about a nominal pose.

Keyboard
--------
``j``  joint-space impedance (spring–damper on ``q_nom - q``)
``t``  SE(3) task-space impedance at ``task_frame`` (default: ``palm``)
``p`` / ``q``  pause / quit

Nominal: YAML ``default_q``; task nominal ``T_nom = FK(task_frame, default_q)``.
Viser publishes gain / mode overrides on ``sw_imp_gains_cmd``.
"""

from __future__ import annotations

import threading

import numpy as np
import pinocchio as pin
from omegaconf import DictConfig

from controller.BaseController import BaseController
from data_type.basic_types.ImpGainsCmdData import ImpGainsCmdData
from data_type.basic_types.ImpStatusData import MODE_JOINT, MODE_TASK, ImpStatusData
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData
from utils.pinocchio.getter import get_fk_link_pose


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion ``(w, x, y, z)``."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


class ImpedanceController(BaseController):
    """Regulate about a fixed nominal with joint or SE(3) task impedance."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._dict_joints = config["dict_joints"]
        self._n = int(config["num_joints"])
        self._num_arm = int(config.get("num_arm_joints", 7))
        self._task_frame = str(config.get("task_frame", "palm"))
        self._gains_cmd_channel = config["sub_manager"].get("gains_cmd_channel", "")
        self._status_channel = config["pub_manager"].get(
            "status_channel", "sw_imp_status"
        )
        self._status_publish_stride = int(config.get("status_publish_stride", 20))
        self._status_tick = 0

        pin_robot = pin.RobotWrapper.BuildFromURDF(
            config["urdf_path"], list(config["mesh_path"].values())
        )
        self._pin_model = pin_robot.model
        self._pin_data = pin_robot.data
        if not self._pin_model.existFrame(self._task_frame):
            raise ValueError(f"task_frame {self._task_frame!r} missing in URDF")
        self._frame_id = self._pin_model.getFrameId(self._task_frame)

        self._q_nom = np.asarray(config["default_q"], dtype=np.float64).copy()
        if self._q_nom.size != self._n:
            raise ValueError("default_q length must equal num_joints")
        self._T_nom = get_fk_link_pose(
            self._q_nom, self._pin_model, self._pin_data, self._task_frame
        )
        self._oMd = pin.SE3(self._T_nom[:3, :3], self._T_nom[:3, 3])

        # Joint impedance (arm); fingers held with soft plant PD to q_nom.
        self._kq_base = np.asarray(config["kq"], dtype=np.float64)
        self._dq_base = np.asarray(config["dq"], dtype=np.float64)
        if self._kq_base.size != self._n or self._dq_base.size != self._n:
            raise ValueError("kq / dq must have length num_joints")

        # SE(3) task stiffness / damping (isotropic trans / rot in body frame).
        self._kt_trans_base = float(config.get("kt_trans", 200.0))
        self._kt_rot_base = float(config.get("kt_rot", 5.0))
        self._dt_trans_base = float(config.get("dt_trans", 20.0))
        self._dt_rot_base = float(config.get("dt_rot", 0.5))

        self._kq_scale = float(config.get("kq_scale", 1.0))
        self._dq_scale = float(config.get("dq_scale", 1.0))
        self._kt_trans_scale = float(config.get("kt_trans_scale", 1.0))
        self._kt_rot_scale = float(config.get("kt_rot_scale", 1.0))
        self._dt_trans_scale = float(config.get("dt_trans_scale", 1.0))
        self._dt_rot_scale = float(config.get("dt_rot_scale", 1.0))

        self._kp_hand = np.asarray(
            config.get("kp_hand", np.zeros(self._n - self._num_arm)),
            dtype=np.float64,
        )
        self._kd_hand = np.asarray(
            config.get("kd_hand", np.zeros(self._n - self._num_arm)),
            dtype=np.float64,
        )
        self._do_grav_comp = bool(config.get("do_grav_comp", True))

        mode0 = str(config.get("mode", "joint")).lower()
        self._mode = "task" if mode0.startswith("t") else "joint"

        self._q = self._q_nom.copy()
        self._qd = np.zeros(self._n, dtype=np.float64)
        self._gains_lock = threading.Lock()
        self._xi = np.zeros(6, dtype=np.float64)

        print(
            f"[Impedance] frame={self._task_frame}, mode={self._mode}. "
            "Keys: j=joint, t=task SE(3), p=pause, q=quit"
        )

    def initialize(self) -> None:
        print(
            f"[Impedance] Initialized. ||T_nom.p||={np.linalg.norm(self._T_nom[:3, 3]):.3f}"
        )

    def _handle_key(self, key: str) -> None:
        if key in ("p", "q"):
            super()._handle_key(key)
            return
        if key == "j":
            self._mode = "joint"
            print("[Impedance] Mode → joint")
            return
        if key == "t":
            self._mode = "task"
            print("[Impedance] Mode → task SE(3)")
            return
        print(f"[Impedance] Unused key: {key!r}")

    def _check_and_get_data_from_que(self) -> None:
        for joint_cfg in self._dict_joints.values():
            channel = joint_cfg["joint_meas_channel"]
            idx = joint_cfg["list_joint_idx"]
            if self.intr_sub_que_dict[channel].empty():
                continue
            meas = self.intr_sub_que_dict[channel].get()
            if not isinstance(meas, JointMeasData):
                raise ValueError(f"Expected JointMeasData on '{channel}'")
            _, q, qd, _ = meas.get_data()
            self._q[idx] = q
            self._qd[idx] = qd

    def _poll_gains_cmd(self) -> None:
        ch = self._gains_cmd_channel
        if not ch or ch not in self.intr_sub_que_dict:
            return
        que = self.intr_sub_que_dict[ch]
        cmd = None
        while not que.empty():
            cmd = que.get()
        if not isinstance(cmd, ImpGainsCmdData):
            return
        with self._gains_lock:
            self._kq_scale = float(cmd.kq_scale)
            self._dq_scale = float(cmd.dq_scale)
            self._kt_trans_scale = float(cmd.kt_trans_scale)
            self._kt_rot_scale = float(cmd.kt_rot_scale)
            self._dt_trans_scale = float(cmd.dt_trans_scale)
            self._dt_rot_scale = float(cmd.dt_rot_scale)
            if int(cmd.mode) == MODE_JOINT:
                self._mode = "joint"
            elif int(cmd.mode) == MODE_TASK:
                self._mode = "task"

    def _effective_joint_gains(self):
        with self._gains_lock:
            return (
                self._kq_base * self._kq_scale,
                self._dq_base * self._dq_scale,
                self._kq_scale,
                self._dq_scale,
            )

    def _effective_task_gains(self):
        with self._gains_lock:
            kt_t = self._kt_trans_base * self._kt_trans_scale
            kt_r = self._kt_rot_base * self._kt_rot_scale
            dt_t = self._dt_trans_base * self._dt_trans_scale
            dt_r = self._dt_rot_base * self._dt_rot_scale
            return (
                kt_t,
                kt_r,
                dt_t,
                dt_r,
                self._kt_trans_scale,
                self._kt_rot_scale,
                self._dt_trans_scale,
                self._dt_rot_scale,
            )

    def _tau_joint_impedance(self, q, qd, kq, dq) -> np.ndarray:
        """τ = Kq (q_nom − q) − Dq q̇  (arm + any joints with nonzero Kq)."""
        return kq * (self._q_nom - q) - dq * qd

    def _tau_task_impedance(self, q, qd, kt_t, kt_r, dt_t, dt_r) -> tuple[np.ndarray, np.ndarray]:
        """SE(3) body impedance at ``task_frame``.

        Error (Pinocchio layout ``(ν, ω)``)::

            ξ = log(T^{-1} T_nom)^∨

        Body twist ``V`` and Jacobian in ``LOCAL``::

            F = K ξ − D V,   τ = J_b^⊤ F

        with diagonal ``K = diag(kt_t I_3, kt_r I_3)`` (and same for ``D``).
        """
        pin.forwardKinematics(self._pin_model, self._pin_data, q, qd)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        oMf = self._pin_data.oMf[self._frame_id]
        # T^{-1} T_nom → log pulls current frame toward nominal.
        iMd = oMf.actInv(self._oMd)
        xi = pin.log6(iMd).vector.copy()  # (v, w)

        J = pin.computeFrameJacobian(
            self._pin_model,
            self._pin_data,
            q,
            self._frame_id,
            pin.ReferenceFrame.LOCAL,
        )  # 6 x nq, (v; w)
        V = pin.getFrameVelocity(
            self._pin_model,
            self._pin_data,
            self._frame_id,
            pin.ReferenceFrame.LOCAL,
        ).vector.copy()

        K = np.diag(
            [kt_t, kt_t, kt_t, kt_r, kt_r, kt_r],
        )
        D = np.diag(
            [dt_t, dt_t, dt_t, dt_r, dt_r, dt_r],
        )
        F = K @ xi - D @ V
        tau = J.T @ F
        return tau, xi

    def _publish_status(
        self,
        mode_id: int,
        q,
        T,
        xi,
        *,
        kq_scale: float,
        dq_scale: float,
        kt_t: float,
        kt_r: float,
        dt_t: float,
        dt_r: float,
    ) -> None:
        if self.pub_que_dict is None:
            return
        que = self.pub_que_dict.get(self._status_channel)
        if que is None:
            return
        self._status_tick += 1
        if self._status_tick % max(1, self._status_publish_stride) != 0:
            return
        status = ImpStatusData(num_joints=self._n, name=self._status_channel)
        status.set_data(
            float(self._cur_time),
            mode_id,
            q,
            self._q_nom,
            T[:3, 3],
            self._T_nom[:3, 3],
            _rot_to_quat_wxyz(T[:3, :3]),
            _rot_to_quat_wxyz(self._T_nom[:3, :3]),
            xi,
            num_arm=self._num_arm,
            kq_scale=kq_scale,
            dq_scale=dq_scale,
            kt_trans=kt_t,
            kt_rot=kt_r,
            dt_trans=dt_t,
            dt_rot=dt_r,
        )
        try:
            while not que.empty():
                que.get_nowait()
        except Exception:
            pass
        try:
            que.put_nowait(status)
        except Exception:
            pass

    def _update(self) -> None:
        self._poll_gains_cmd()
        q = self._q.copy()
        qd = self._qd.copy()

        kq, dq, kq_s, dq_s = self._effective_joint_gains()
        kt_t, kt_r, dt_t, dt_r, ktt_s, ktr_s, dtt_s, dtr_s = self._effective_task_gains()

        if self._do_grav_comp:
            tau_ff = pin.computeGeneralizedGravity(self._pin_model, self._pin_data, q)
        else:
            tau_ff = np.zeros(self._n, dtype=np.float64)

        xi = np.zeros(6, dtype=np.float64)
        if self._mode == "task":
            tau_imp, xi = self._tau_task_impedance(q, qd, kt_t, kt_r, dt_t, dt_r)
            tau_ff = tau_ff + tau_imp
            mode_id = MODE_TASK
            # Effective displayed stiffness already includes scale.
            kt_disp, kr_disp = kt_t, kt_r
            dt_disp, dr_disp = dt_t, dt_r
        else:
            tau_ff = tau_ff + self._tau_joint_impedance(q, qd, kq, dq)
            mode_id = MODE_JOINT
            kt_disp = self._kt_trans_base * ktt_s
            kr_disp = self._kt_rot_base * ktr_s
            dt_disp = self._dt_trans_base * dtt_s
            dr_disp = self._dt_rot_base * dtr_s
            # Joint-space pose error as SE(3) for viz only.
            T_cur = get_fk_link_pose(q, self._pin_model, self._pin_data, self._task_frame)
            oMf = pin.SE3(T_cur[:3, :3], T_cur[:3, 3])
            xi = pin.log6(oMf.actInv(self._oMd)).vector.copy()

        self._xi = xi

        # Arm impedance lives in tau_ff; soft PD keeps the hand at q_nom.
        kp = np.zeros(self._n, dtype=np.float64)
        kd = np.zeros(self._n, dtype=np.float64)
        hand = slice(self._num_arm, self._n)
        n_hand = self._n - self._num_arm
        if self._kp_hand.size == n_hand:
            kp[hand] = self._kp_hand
            kd[hand] = self._kd_hand
        elif self._kp_hand.size == self._n:
            kp[hand] = self._kp_hand[hand]
            kd[hand] = self._kd_hand[hand]

        cmd = JointCtrlData(num_joints=self._n)
        cmd.set_data(
            self._cur_time,
            self._q_nom.copy(),
            np.zeros(self._n, dtype=np.float64),
            tau_ff,
            kp,
            kd,
        )
        self.ctrl_pub_que.put(cmd)

        T = get_fk_link_pose(q, self._pin_model, self._pin_data, self._task_frame)
        self._publish_status(
            mode_id,
            q,
            T,
            xi,
            kq_scale=kq_s,
            dq_scale=dq_s,
            kt_t=kt_disp,
            kt_r=kr_disp,
            dt_t=dt_disp,
            dt_r=dr_disp,
        )
