import numpy as np
import pinocchio as pin
from pinocchio import FrameType, ReferenceFrame
import mujoco

import copy

from termcolor import colored
import numpy as np

import time


def highlight_diff(val, threshold=1e-4, label="difference"):
    """Print value in bold red if above threshold."""
    if val > threshold:
        print(colored(f"{label}: {val:.4e}", "red", attrs=["bold"]))
    else:
        print(f"{label}: {val:.4e}")


class ModelComparator:
    def __init__(self, urdf_path, xml_path, mesh_paths):
        """
        Initialize the ModelComparator with Pinocchio and MuJoCo models and data.
        """
        pin_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_paths)
        self.pin_model = pin_robot.model
        self.pin_data = pin_robot.data

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

    def compare_joint_names(self):
        pin_joint_names = [name for name in self.pin_model.names][1:]  # Skip 'universe'
        mj_joint_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.mj_model.njnt)
        ]
        return pin_joint_names, mj_joint_names

    def compare_link_names(self):
        pin_link_names = [
            frame.name
            for frame in self.pin_model.frames
            if frame.type == FrameType.BODY
        ]
        mj_link_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.mj_model.nbody)
        ]
        return pin_link_names, mj_link_names

    def compare_joint_limits(self):
        pin_joint_lower_limits = self.pin_model.lowerPositionLimit
        pin_joint_upper_limits = self.pin_model.upperPositionLimit
        pin_joint_limits = []

        for ll, ul in zip(pin_joint_lower_limits, pin_joint_upper_limits):
            pin_joint_limits.append((ll, ul))

        mj_joint_limits = [
            (self.mj_model.jnt_range[i][0], self.mj_model.jnt_range[i][1])
            for i in range(self.mj_model.njnt)
        ]
        return pin_joint_limits, mj_joint_limits

    def compare_joint_velocity_limits(self):
        pin_joint_velocity_limits = self.pin_model.velocityLimit
        mj_joint_velocity_limits = [None for i in range(self.mj_model.njnt)]
        return pin_joint_velocity_limits, mj_joint_velocity_limits

    def compare_effort_ctrl_limits(self):
        pin_joint_effort_limits = self.pin_model.effortLimit
        mj_joint_ctrl_limits = [
            self.mj_model.actuator_ctrlrange[i] for i in range(self.mj_model.njnt)
        ]
        return pin_joint_effort_limits, mj_joint_ctrl_limits

    def compare_forward_kinematics(self, q):
        # --- Pinocchio Forward Kinematics ---
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        pin_poses = {}
        for frame_id, frame in enumerate(self.pin_model.frames):
            if frame.type == FrameType.BODY:
                pose = self.pin_data.oMf[frame_id]  # SE3
                pin_poses[frame.name] = (pose.translation.copy(), pose.rotation.copy())

        # --- MuJoCo Forward Kinematics ---
        self.mj_data.qpos[:] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)

        mj_poses = {}
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            pos = self.mj_data.xpos[i].copy()
            rot = self.mj_data.xmat[i].reshape(3, 3).copy()
            mj_poses[name] = (pos, rot)
        return pin_poses, mj_poses

    def compare_jacobian(self, q):
        # --- Pinocchio: Compute Jacobians at each BODY frame ---
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)

        pin_jacobians = {}
        for frame_id, frame in enumerate(self.pin_model.frames):
            if frame.type == FrameType.BODY:
                J = pin.getFrameJacobian(
                    self.pin_model,
                    self.pin_data,
                    frame_id,
                    ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                pin_jacobians[frame.name] = J.copy()

        # --- MuJoCo: Compute Jacobians for each body ---
        self.mj_data.qpos[:] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)

        nv = self.mj_model.nv
        mj_jacobians = {}

        for body_id in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            J_pos = np.zeros((3, nv))
            J_rot = np.zeros((3, nv))
            mujoco.mj_jacBody(self.mj_model, self.mj_data, J_pos, J_rot, body_id)

            # Stack into 6xN Jacobian (Pinocchio format)
            J = np.vstack((J_pos, J_rot))
            mj_jacobians[name] = J
        return pin_jacobians, mj_jacobians

    def compare_gravity(self, q):
        """
        Compare gravity compensation torques from Pinocchio and MuJoCo
        at the given joint configuration q.
        """

        # --- Pinocchio: Compute Gravity Vector ---
        pin_g = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)

        # --- MuJoCo: Compute Gravity Vector ---
        self.mj_data.qpos[:] = copy.deepcopy(q)
        self.mj_data.qvel[:] = 0.0  # zero velocity
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Note: mj_rnePostConstraint sets qfrc_bias = gravity + Coriolis + centrifugal
        # Since qvel=0, Coriolis = 0 → qfrc_bias = gravity
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)

        mj_g = copy.deepcopy(self.mj_data.qfrc_bias)

        return pin_g, mj_g

    def compare_gravity_plus_coriolis(self, q, qd):
        """
        Compare bias generalized forces (gravity + coriolis/centrifugal):
          Pinocchio: data.nle
          MuJoCo:    data.qfrc_bias
        """

        # --- Pinocchio bias: C(q,qd) + g(q) ---
        pin.computeAllTerms(self.pin_model, self.pin_data, q, qd)
        pin_bias = self.pin_data.nle.copy()

        # --- MuJoCo bias: C(q,qd) + g(q) ---
        self.mj_data.qpos[:] = q
        self.mj_data.qvel[:] = qd
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mj_bias = self.mj_data.qfrc_bias.copy()

        return pin_bias, mj_bias

    def compare_mass_matrix(self, q):
        # --- Pinocchio: Compute Mass Matrix ---
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.crba(self.pin_model, self.pin_data, q)
        pin_mass_matrix = self.pin_data.M.copy()

        # --- MuJoCo: Compute Mass Matrix ---
        self.mj_data.qpos[:] = copy.deepcopy(q)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        nv = self.mj_model.nv
        mj_mass_matrix = np.zeros((nv, nv))
        mujoco.mj_fullM(self.mj_model, mj_mass_matrix, self.mj_data.qM)

        return pin_mass_matrix, mj_mass_matrix

    def compare_inverse_dynamics(self, q, qd, qdd):
        """
        Compare inverse dynamics torques from Pinocchio and MuJoCo
        at the given joint configuration q, velocity qd, and acceleration qdd.
        """

        # ---- Pinocchio (unconstrained inverse dynamics) ----
        pin_tau = pin.rnea(self.pin_model, self.pin_data, q, qd, qdd)

        # ---- MuJoCo setup ----
        q = np.asarray(q).copy()
        qd = np.asarray(qd).copy()
        qdd = np.asarray(qdd).copy()

        # Sanity checks: MuJoCo uses nq for qpos, nv for qvel/qacc/qfrc_*
        assert q.shape[0] == self.mj_model.nq, "q must match mj_model.nq"
        assert qd.shape[0] == self.mj_model.nv, "qd must match mj_model.nv"
        assert qdd.shape[0] == self.mj_model.nv, "qdd must match mj_model.nv"

        self.mj_data.qpos[:] = q
        self.mj_data.qvel[:] = qd
        self.mj_data.qacc[:] = qdd

        # Controls do not matter for qfrc_inverse, but keep deterministic if you want
        if self.mj_model.nu > 0:
            self.mj_data.ctrl[:] = 0.0

        # Run forward kinematics + dynamics terms (computes qM, qfrc_bias, etc.)
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Compute inverse dynamics at (q, qd, qdd)
        mujoco.mj_inverse(self.mj_model, self.mj_data)

        mj_tau_full = self.mj_data.qfrc_inverse.copy()

        return pin_tau, mj_tau_full

    def compare_computation_time(self, n_times=1000):
        """
        Compare computation time of Pinocchio and MuJoCo
        """

        # Forward kinematics computation
        # Generate random joint configuration
        q = np.random.uniform(-1, 1, self.pin_model.nq)

        # Pinocchio
        start_time = time.time()
        for _ in range(n_times):
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin_time = (time.time() - start_time) / n_times
        print(f"Pinocchio forward kinematics time: {pin_time:.4e} seconds")

        # MuJoCo
        start_time = time.time()
        for _ in range(n_times):
            self.mj_data.qpos[:] = q
            mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco_time = (time.time() - start_time) / n_times
        print(f"MuJoCo forward kinematics time: {mujoco_time:.4e} seconds")

        # Jacobian computation
        # Pinocchio
        start_time = time.time()
        for _ in range(n_times):
            pin.computeJointJacobians(self.pin_model, self.pin_data, q)
            pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                0,  # Assuming we want the Jacobian of the first frame
                ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
        pin_jacobian_time = (time.time() - start_time) / n_times
        print(f"Pinocchio Jacobian computation time: {pin_jacobian_time:.4e} seconds")
        # MuJoCo
        start_time = time.time()
        for _ in range(n_times):
            self.mj_data.qpos[:] = q
            mujoco.mj_forward(self.mj_model, self.mj_data)
            mujoco.mj_jacBody(self.mj_model, self.mj_data, None, None, 0)
        mujoco_jacobian_time = (time.time() - start_time) / n_times
        print(f"MuJoCo Jacobian computation time: {mujoco_jacobian_time:.4e} seconds")

        # Gravity computation
        # Pinocchio
        start_time = time.time()
        for _ in range(n_times):
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        pin_gravity_time = (time.time() - start_time) / n_times
        print(f"Pinocchio gravity computation time: {pin_gravity_time:.4e} seconds")
        # MuJoCo
        start_time = time.time()
        for _ in range(n_times):
            self.mj_data.qpos[:] = q
            self.mj_data.qvel[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco_gravity_time = (time.time() - start_time) / n_times
        print(f"MuJoCo gravity computation time: {mujoco_gravity_time:.4e} seconds")

        # Coriolis + Gravity computation
        # Pinocchio
        start_time = time.time()
        for _ in range(n_times):
            pin.computeAllTerms(self.pin_model, self.pin_data, q, np.zeros_like(q))
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        pin_coriolis_time = (time.time() - start_time) / n_times
        print(
            f"Pinocchio Coriolis + Gravity computation time: {pin_coriolis_time:.4e} seconds"
        )
        # MuJoCo
        start_time = time.time()
        for _ in range(n_times):
            self.mj_data.qpos[:] = q
            self.mj_data.qvel[:] = np.zeros_like(q)
            mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco_coriolis_time = (time.time() - start_time) / n_times
        print(
            f"MuJoCo Coriolis + Gravity computation time: {mujoco_coriolis_time:.4e} seconds"
        )

    def compare(self):
        # ===============================
        # [1] Compare Joint Names
        # ===============================
        print("\n[1] Compare joint names")
        pin_joints, mj_joints = self.compare_joint_names()
        print("Pinocchio joints:", pin_joints)
        print("MuJoCo joints:   ", mj_joints)

        # ===============================
        # [2] Compare Link Names
        # ===============================
        print("\n[2] Compare link names")
        pin_links, mj_links = self.compare_link_names()
        print("Pinocchio links:", pin_links)
        print("MuJoCo links:   ", mj_links)

        # ===============================
        # [3] Compare Joint Limits
        # ===============================
        print("\n[3] Compare joint limits")
        pin_limits, mj_limits = self.compare_joint_limits()
        for i, (pin_limit, mj_limit) in enumerate(zip(pin_limits, mj_limits)):
            print(f"Joint {i}: Pinocchio: {pin_limit}, MuJoCo: {mj_limit}")

        # ===============================
        # [4] Compare Joint Velocity Limits
        # ===============================
        print("\n[4] Compare joint velocity limits")
        pin_velocity_limits, mj_velocity_limits = self.compare_joint_velocity_limits()
        for i, (pin_limit, mj_limit) in enumerate(
            zip(pin_velocity_limits, mj_velocity_limits)
        ):
            print(f"Joint {i}: Pinocchio: {pin_limit}, MuJoCo: {mj_limit}")

        # ===============================
        # [5] Compare Effort / Control Limits
        # ===============================
        print("\n[5] Compare effort and control limits")
        pin_effort_limits, mj_ctrl_limits = self.compare_effort_ctrl_limits()
        for i, (pin_limit, mj_limit) in enumerate(
            zip(pin_effort_limits, mj_ctrl_limits)
        ):
            print(f"Joint {i}: Pinocchio: {pin_limit}, MuJoCo: {mj_limit}")

        # ===============================
        # [6] Compare Forward Kinematics
        # ===============================
        print("\n[6] Compare forward kinematics")
        q = np.random.uniform(-3, 3, self.pin_model.nq)
        pin_poses, mj_poses = self.compare_forward_kinematics(q)
        for name in pin_poses:
            pin_pos, pin_rot = pin_poses[name]
            mj_pos, mj_rot = mj_poses[name]
            pos_diff = np.linalg.norm(pin_pos - mj_pos)
            rot_diff = np.linalg.norm(pin_rot - mj_rot)
            print(f"Link {name}:")
            print(f"  Pinocchio pos: {pin_pos}, rot: {pin_rot}")
            print(f"  MuJoCo    pos: {mj_pos}, rot: {mj_rot}")
            highlight_diff(pos_diff, label="  Position difference")
            highlight_diff(rot_diff, label="  Rotation difference")

        # ===============================
        # [7] Compare Jacobians
        # ===============================
        print("\n[7] Compare Jacobians")
        q = np.random.uniform(-1, 1, self.pin_model.nq)
        pin_jacobians, mj_jacobians = self.compare_jacobian(q=q)
        for name in pin_jacobians:
            pin_J = pin_jacobians[name]
            mj_J = mj_jacobians[name]
            diff = np.linalg.norm(pin_J - mj_J)
            print(f"Link {name} Jacobian difference:")
            highlight_diff(diff, label="  Jacobian difference")

        # ===============================
        # [8] Compare Gravity Terms
        # ===============================
        print("\n[8] Compare gravity terms")
        q = np.random.uniform(-1, 1, self.pin_model.nq)
        pin_g, mj_g = self.compare_gravity(q=q)
        print(f"Pinocchio gravity: {pin_g}")
        print(f"MuJoCo gravity:    {mj_g}")
        highlight_diff(np.linalg.norm(pin_g - mj_g), label="Gravity difference")

        # ===============================
        # [9] Compare Coriolis + Gravity Terms
        # ===============================
        print("\n[9] Compare Coriolis + Gravity terms")
        q = np.random.uniform(-1, 1, self.pin_model.nq)
        qd = np.random.uniform(-1, 1, self.pin_model.nv)
        pin_coriolis, mj_coriolis = self.compare_gravity_plus_coriolis(q=q, qd=qd)
        print(f"Pinocchio gravity + Coriolis: {pin_coriolis}")
        print(f"MuJoCo gravity + Coriolis:    {mj_coriolis}")
        highlight_diff(
            np.linalg.norm(pin_coriolis - mj_coriolis), label="Coriolis difference"
        )

        # ===============================
        # [10] Compare Mass Matrices
        # ===============================
        print("\n[10] Compare mass matrices")
        q = np.random.uniform(-1, 1, self.pin_model.nq)
        pin_mass_matrix, mj_mass_matrix = self.compare_mass_matrix(q=q)

        # print(f"Pinocchio mass matrix:\n{pin_mass_matrix}")
        # print(f"MuJoCo mass matrix:\n{mj_mass_matrix}")
        highlight_diff(
            np.linalg.norm(pin_mass_matrix - mj_mass_matrix),
            label="Mass matrix difference",
        )

        # ===============================
        # [11] Compare Inverse Dynamics
        # ===============================
        print("\n[11] Compare inverse dynamics")
        q = np.random.uniform(-1, 1, self.pin_model.nq)
        qd = np.random.uniform(-1, 1, self.pin_model.nv)
        qdd = np.random.uniform(-1, 1, self.pin_model.nv)
        pin_torques, mj_torques = self.compare_inverse_dynamics(q, qd, qdd)
        print(f"Pinocchio torques: {pin_torques}")
        print(f"MuJoCo torques:    {mj_torques}")
        highlight_diff(
            np.linalg.norm(pin_torques - mj_torques),
            label="Inverse dynamics difference",
        )

        # ===============================
        # [12] MUJOCO QFRC ...
        # ===============================
        print("\n[12] MuJoCo qfrc_inverse, qfrc_bias, qfrc_constraint, ...")

        # Random state
        # sample q from joint limits

        q = np.array(
            [
                1.0,
                -0.3,
                0.0,
                -1.57,
                -0.5,
                0.0,
                0.0,
                -1.0,
                -1.5,
                0.5,
                0.1,
                -0.3,
                1.2,
                0.7,
                0.5,
                0.0,
                1.2,
                0.8,
                0.5,
                0.4,
                1.2,
                0.9,
                0.5,
                0.6,
                1.2,
                1.0,
                0.5,
            ]
        )  # no collision

        qd = np.random.uniform(-1, 1, self.pin_model.nv) * 5
        qdd = np.random.uniform(-1, 1, self.pin_model.nv) * 5

        self.mj_data.qpos[:] = q
        self.mj_data.qvel[:] = qd
        self.mj_data.qacc[:] = qdd

        # IMPORTANT: make sure controls are defined if your model has actuators
        if self.mj_model.nu > 0:
            self.mj_data.ctrl[:] = 0.0

        # Run forward kinematics + dynamics terms (computes qM, qfrc_bias, etc.)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # ---------
        # Bias term: C(q,qd)+G(q)
        # ---------
        qfrc_bias = self.mj_data.qfrc_bias.copy()

        # ---------
        # Mass-matrix times qdd: M(q) qdd
        # Use mj_mulM (preferred; no need to form dense M)
        # ---------
        Mqdd = np.zeros(self.mj_model.nv)
        mujoco.mj_mulM(self.mj_model, self.mj_data, Mqdd, self.mj_data.qacc)

        # LHS dynamics term (what textbooks call "inertial + coriolis + gravity")
        lhs = Mqdd + qfrc_bias

        # ---------
        # RHS forces (what the simulator applies)
        # ---------
        qfrc_actuator = self.mj_data.qfrc_actuator.copy()
        qfrc_passive = self.mj_data.qfrc_passive.copy()
        qfrc_applied = self.mj_data.qfrc_applied.copy()
        qfrc_constraint = self.mj_data.qfrc_constraint.copy()

        rhs = qfrc_actuator + qfrc_passive + qfrc_applied + qfrc_constraint

        # ---------
        # Optional: inverse dynamics (RNE-style) torque
        # NOTE: qfrc_inverse is typically computed by mj_inverse / mj_rnePostConstraint.
        # Use mj_inverse for a clean "no constraint forces" inverse dynamics.
        # ---------
        mujoco.mj_inverse(self.mj_model, self.mj_data)
        qfrc_inverse = self.mj_data.qfrc_inverse.copy()

        # Print
        np.set_printoptions(precision=4, suppress=True)

        print("nv:", self.mj_model.nv, "nq:", self.mj_model.nq, "nu:", self.mj_model.nu)
        print("||qfrc_bias||       =", np.linalg.norm(qfrc_bias))
        print("||Mqdd||            =", np.linalg.norm(Mqdd))
        print("||lhs=Mqdd+bias||   =", np.linalg.norm(lhs))

        print("||qfrc_actuator||   =", np.linalg.norm(qfrc_actuator))
        print("||qfrc_passive||    =", np.linalg.norm(qfrc_passive))
        print("||qfrc_applied||    =", np.linalg.norm(qfrc_applied))
        print("||qfrc_constraint|| =", np.linalg.norm(qfrc_constraint))
        print("||rhs sum||         =", np.linalg.norm(rhs))

        print("||lhs-rhs||         =", np.linalg.norm(lhs - rhs))
        print("||qfrc_inverse||    =", np.linalg.norm(qfrc_inverse))
