import os
import numpy as np
import trimesh

from utils.pinocchio.getter import get_link_Jacobian, get_fk_link_pose
from utils.lie.kinematics import get_point_Jacobian

import scipy.sparse as sparse
import osqp

from utils.grasping.force_gen import (
    generate_contact_forces,
    compute_contact_frame,
)


class GraspSqueezeParamsGenerator:
    def __init__(self, pin_robot, mesh, grasp_data, config):
        self.pin_robot = pin_robot
        self.mesh = mesh
        self.grasp_data = grasp_data
        self.config = config or {}
        self._device = config.get("device", "cpu")

        self._pin_model = pin_robot.model
        self._pin_data = pin_robot.data

        self._torque_limit = np.array(
            self.config.get("torque_limit", [0.2] * self._pin_model.nq)
        )  # rated torque limit for each joint

        self._q_lower_limit = self._pin_model.lowerPositionLimit
        self._q_upper_limit = self._pin_model.upperPositionLimit

        self._contact_points_in_link_frame = (
            grasp_data.get_contact_points_in_link_frame()
        )
        self._link_idxs_of_contact_points = grasp_data.get_link_idxs_of_contact_points()
        self._contact_points_in_cad_frame = grasp_data.get_contact_points_in_cad_frame()
        self._contact_normals_in_cad_frame = (
            grasp_data.get_contact_normals_in_cad_frame()
        )

        self._mu = self.config.get("mu", 0.3)
        self._min_fn = self.config.get("min_fn", 1.0)

        # Hand-contact baseline: use FK-derived hand-surface points/normals as
        # the contact geometry instead of mesh-projected ones. Robust to
        # object-pose error since hand FK is independent of perceived object pose.
        self._use_hand_contacts = self.config.get("use_hand_contacts", False)
        if self._use_hand_contacts:
            self._overwrite_contacts_with_hand_fk()

    def get_contact_points_in_link_frame(self):
        return self._contact_points_in_link_frame

    def get_link_idxs_of_contact_points(self):
        return self._link_idxs_of_contact_points

    def get_contact_points_in_cad_frame(self):
        return self._contact_points_in_cad_frame

    def get_contact_normals_in_cad_frame(self):
        return self._contact_normals_in_cad_frame

    def set_contact_points_in_link_frame(self, contact_points_in_link_frame):
        self._contact_points_in_link_frame = contact_points_in_link_frame

    def set_link_idxs_of_contact_points(self, link_idxs_of_contact_points):
        self._link_idxs_of_contact_points = link_idxs_of_contact_points

    def set_contact_points_in_cad_frame(self, contact_points_in_cad_frame):
        self._contact_points_in_cad_frame = contact_points_in_cad_frame

    def set_contact_normals_in_cad_frame(self, contact_normals_in_cad_frame):
        self._contact_normals_in_cad_frame = contact_normals_in_cad_frame

    def _overwrite_contacts_with_hand_fk(self):
        """Replace mesh-projected contact points/normals (in CAD frame) with
        hand-FK-derived ones. Independent of perceived object pose.

        Convention: stored normals point outward from the hand (toward the
        object) — i.e., the push direction. ``generate_contact_forces`` accounts
        for this when feeding ``compute_contact_frame``.
        """
        normals_link = self.grasp_data.get_contact_normals_in_link_frame()
        if normals_link is None:
            print(
                "[GraspSqueezeParamsGenerator] WARNING: use_hand_contacts is "
                "set but contact_normals_in_link_frame.npy is missing from the "
                "grasp_predictions folder. Falling back to mesh-projected "
                "offline contacts (real-time control unaffected if it uses "
                "contact_source: hand)."
            )
            self._use_hand_contacts = False  # disable offline override
            return

        grasp_poses = self.grasp_data.get_grasp_poses_in_cad_frame()
        T_grasp_in_palm = self.grasp_data.get_SE3_grasp_pose_in_palm_frame()
        T_palm_in_grasp = np.linalg.inv(T_grasp_in_palm)
        num_contacts = self.grasp_data.get_num_contacts_per_grasp()
        mapper = self.grasp_data.get_mapper_idx_to_link_name()

        N, P_max, _ = self._contact_points_in_link_frame.shape
        points_cad = np.zeros((N, P_max, 3))
        normals_cad = np.zeros((N, P_max, 3))

        for i in range(N):
            P_i = int(num_contacts[i])
            joint_angles = grasp_poses[i, :20]

            T_grasp_in_cad = np.eye(4)
            T_grasp_in_cad[:3, 3] = grasp_poses[i, 20:23]
            T_grasp_in_cad[:3, :3] = grasp_poses[i, 23:32].reshape(3, 3)
            T_palm_in_cad = T_grasp_in_cad @ T_palm_in_grasp

            for j in range(P_i):
                link_name = mapper[self._link_idxs_of_contact_points[i, j]]
                T_link_in_palm = get_fk_link_pose(
                    joint_angles, self._pin_model, self._pin_data, link_name,
                )
                T_link_in_cad = T_palm_in_cad @ T_link_in_palm
                R = T_link_in_cad[:3, :3]
                t = T_link_in_cad[:3, 3]
                points_cad[i, j] = R @ self._contact_points_in_link_frame[i, j] + t
                normals_cad[i, j] = R @ normals_link[i, j]

        self._contact_points_in_cad_frame = points_cad
        self._contact_normals_in_cad_frame = normals_cad

    def generate_contact_forces(self):
        #####################
        # prepare varialbes #
        #####################
        null_space_margin = self.config.get("null_space_margin", 0.01)
        N = self._contact_points_in_link_frame.shape[0]  # number of grasp poses
        P_max = self._contact_points_in_link_frame.shape[1]  # max contact points
        nq = self._pin_model.nq  # number of joints
        num_contacts = self.grasp_data.get_num_contacts_per_grasp()  # (N,)

        grasp_poses = self.grasp_data.get_grasp_poses_in_cad_frame()

        f_cad = np.zeros((N, P_max, 3))
        for i in range(N):
            P_i = int(num_contacts[i])
            print(f"Generating contact forces for grasp pose {i}/{N} (P={P_i})")

            # Compute Jacobians for valid contacts only
            Jr_i = np.zeros((P_i, 3, nq))
            for j in range(P_i):
                link_name = self.grasp_data.get_mapper_idx_to_link_name()[
                    self._link_idxs_of_contact_points[i, j]
                ]
                T = get_fk_link_pose(
                    grasp_poses[i, :20],
                    self._pin_model,
                    self._pin_data,
                    link_name,
                )
                J = get_link_Jacobian(
                    grasp_poses[i, :20],
                    self._pin_model,
                    self._pin_data,
                    link_name,
                )
                Jr_i[j, :, :] = get_point_Jacobian(
                    T[0:3, 0:3],
                    J[3:, :],
                    J[:3, :],
                    self._contact_points_in_link_frame[i, j, :],
                )

            c_i = self._contact_points_in_cad_frame[i, :P_i, :]  # (P_i, 3)
            # compute_contact_frame wants n in the push direction (into the object).
            # Mesh normals point outward from the object → negate.
            # Hand normals point outward from the hand (toward object) → already push dir.
            if self._use_hand_contacts:
                n_i = self._contact_normals_in_cad_frame[i, :P_i, :]
            else:
                n_i = -self._contact_normals_in_cad_frame[i, :P_i, :]

            # compute_contact_frame expects (N, P, 3, ...) — use batch dim of 1
            Jo_i, Jr_contact_i, n_i, t1_i, t2_i = compute_contact_frame(
                n_i[np.newaxis], c_i[np.newaxis], Jr_i[np.newaxis]
            )

            f_i, solved = generate_contact_forces(
                Jo_i[0],  # (P_i, 3, 6)
                Jr_contact_i[0],  # (P_i, 3, nq)
                self._torque_limit,
                self._min_fn,
                self._mu,
                null_space_margin,
            )  # (P_i, 3) in contact frame [fn, ft1, ft2]

            # coordinate transformation from [fn, ft1, ft2] to [fx, fy, fz] in cad frame
            f_cad[i, :P_i, :] = (
                f_i[:, 0:1] * n_i[0] + f_i[:, 1:2] * t1_i[0] + f_i[:, 2:3] * t2_i[0]
            )

        return f_cad
