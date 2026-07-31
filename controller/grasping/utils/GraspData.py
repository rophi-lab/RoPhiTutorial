import numpy as np
import os


class GraspData:
    def __init__(self, grasp_predictions_path):
        self._load_data_from_file(grasp_predictions_path)

        self._mapper_idx_to_link_name = None  # To be overridden by subclasses
        self._link_names_of_contact_points = None  # To be overridden by subclasses

    def _set_link_names_of_contact_points(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    ## GETTERS
    def get_grasp_poses_in_cad_frame(self):
        return self._grasp_poses_in_cad_frame

    def get_SE3_grasp_pose_in_palm_frame(self):
        return self._SE3_grasp_pose_in_palm_frame

    def get_link_names_of_contact_points(self):
        return self._link_names_of_contact_points

    def get_contact_points_in_link_frame(self):
        return self._contact_points_in_link_frame

    def get_contact_normals_in_link_frame(self):
        # May be None for older grasp_predictions folders without this file.
        return self._contact_normals_in_link_frame

    def get_contact_points_in_cad_frame(self):
        return self._contact_points_in_cad_frame

    def get_contact_normals_in_cad_frame(self):
        return self._contact_normals_in_cad_frame

    def get_link_idxs_of_contact_points(self):
        return self._link_idxs_of_contact_points

    def get_mapper_idx_to_link_name(self):
        return self._mapper_idx_to_link_name

    def get_pregrasp_offset_q_gripper(self):
        return self._pregrasp_offset_q_gripper

    def get_pregrasp_palm_offset_in_palm_frame(self):
        return self._pregrasp_palm_offset_in_palm_frame

    def get_num_contacts_per_grasp(self):
        return self._num_contacts_per_grasp

    ## PRIVATE METHODS
    def _load_data_from_file(self, grasp_predictions_path):
        self._grasp_poses_in_cad_frame = np.load(
            os.path.join(grasp_predictions_path, "grasp_poses.npy")
        )
        # This should be loaded from the file and numpy of shape (N, 20 + 3 + 9)
        # 20 : 20 joints of hand
        # 3 : 3 DoF of the grasp position
        # 9 : 9 DoF of the grasp rotation_matrix.flatten()
        self._SE3_grasp_pose_in_palm_frame = np.load(
            os.path.join(
                grasp_predictions_path,
                "SE3_grasp_pose_in_palm_frame.npy",
            )
        )  # (4, 4) SE3 grasp pose in palm frame -- basically constant offset from the palm frame to SE3 of generated grasp poses

        # set contact points
        self._set_contact_points(grasp_predictions_path)

        pregrasp_path = os.path.join(
            grasp_predictions_path, "pregrasp_offset_q_gripper.npy"
        )
        if os.path.exists(pregrasp_path):
            self._pregrasp_offset_q_gripper = np.load(pregrasp_path)
        else:
            self._pregrasp_offset_q_gripper = np.zeros(
                (self._grasp_poses_in_cad_frame.shape[0], 20)
            )

        # Per-grasp palm pregrasp offset, expressed in the palm-LOCAL frame.
        # The convention (matches PregraspParamsGenerator + ViserMeshGraspVisualizer):
        #   T_palm_pregrasp = T_palm_grasp @ translate(palm_offset_local)
        # where translate(.) is a pure-translation SE3. Zero if the file is
        # missing (backwards-compatible with older grasp prediction folders).
        pregrasp_palm_offset_path = os.path.join(
            grasp_predictions_path, "pregrasp_palm_offset_in_palm_frame.npy"
        )
        if os.path.exists(pregrasp_palm_offset_path):
            self._pregrasp_palm_offset_in_palm_frame = np.load(
                pregrasp_palm_offset_path
            )
        else:
            self._pregrasp_palm_offset_in_palm_frame = np.zeros(
                (self._grasp_poses_in_cad_frame.shape[0], 3)
            )

    def _set_contact_points(self, grasp_predictions_path):
        # Load contact points information from the file
        # num_contacts = 12  # only active contacts are considered
        contact_points_path = os.path.join(grasp_predictions_path, "contact_points.npy")
        contact_normals_path = os.path.join(
            grasp_predictions_path, "contact_normals.npy"
        )
        contact_link_indices_path = os.path.join(
            grasp_predictions_path, "contact_link_indices.npy"
        )
        contact_points_in_link_frame_path = os.path.join(
            grasp_predictions_path, "contact_points_in_link_frame.npy"
        )
        assert os.path.exists(
            contact_points_path
        ), f"Contact points file not found at {contact_points_path}"
        assert os.path.exists(
            contact_normals_path
        ), f"Contact normals file not found at {contact_normals_path}"
        assert os.path.exists(
            contact_link_indices_path
        ), f"Contact link indices file not found at {contact_link_indices_path}"
        assert os.path.exists(
            contact_points_in_link_frame_path
        ), f"Contact points in link frame file not found at {contact_points_in_link_frame_path}"

        contact_points_in_cad_frame = np.load(contact_points_path)  # (N, P_max, 3)
        contact_normals_in_cad_frame = np.load(contact_normals_path)  # (N, P_max, 3)

        self._contact_points_in_link_frame = np.load(contact_points_in_link_frame_path)
        self._contact_points_in_cad_frame = contact_points_in_cad_frame
        self._contact_normals_in_cad_frame = contact_normals_in_cad_frame
        self._link_idxs_of_contact_points = np.load(contact_link_indices_path)

        # Optional: hand-link-frame normals (saved by newer GraspQP runs).
        # Required when use_hand_contacts is enabled in GraspSqueezeParamsGenerator;
        # absent in older grasp_predictions folders (None falls through and is
        # only checked when actually needed).
        contact_normals_in_link_frame_path = os.path.join(
            grasp_predictions_path, "contact_normals_in_link_frame.npy"
        )
        if os.path.exists(contact_normals_in_link_frame_path):
            self._contact_normals_in_link_frame = np.load(
                contact_normals_in_link_frame_path
            )
        else:
            self._contact_normals_in_link_frame = None

        # Per-grasp contact count (backward compat: default to P_max if file missing)
        num_contacts_path = os.path.join(grasp_predictions_path, "num_contacts.npy")
        if os.path.exists(num_contacts_path):
            self._num_contacts_per_grasp = np.load(num_contacts_path).astype(np.int64)
        else:
            N = contact_points_in_cad_frame.shape[0]
            P_max = contact_points_in_cad_frame.shape[1]
            self._num_contacts_per_grasp = np.full(N, P_max, dtype=np.int64)

    def _compute_desired_grasp_contact_forces(self):
        pass


# HELPER FUNCTIONS
def convert_grasp_poses_in_A_to_B(grasp_poses_in_A, A2B, num_hand_joints=20):
    grasp_poses_in_B = grasp_poses_in_A.copy()

    p_in_A = grasp_poses_in_A[:, num_hand_joints : num_hand_joints + 3]
    R_in_A = grasp_poses_in_A[:, num_hand_joints + 3 : num_hand_joints + 12].reshape(
        -1, 3, 3
    )
    p_A2B = A2B[:3, 3].reshape(1, 3)
    R_A2B = A2B[:3, :3].reshape(1, 3, 3)
    p_in_B = p_A2B + (R_A2B @ p_in_A.reshape(-1, 3, 1)).reshape(-1, 3)
    R_in_B = R_A2B @ R_in_A

    grasp_poses_in_B[:, num_hand_joints : num_hand_joints + 3] = p_in_B
    grasp_poses_in_B[:, num_hand_joints + 3 : num_hand_joints + 12] = R_in_B.reshape(
        -1, 9
    )
    return grasp_poses_in_B


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--grasp_predictions_path", type=str, required=True)
    args = parser.parse_args()

    grasp_data = GraspData(args.grasp_predictions_path)
    print("Grasp poses in CAD frame:", grasp_data.get_grasp_poses_in_cad_frame().shape)
    print(
        "SE3 grasp pose in palm frame:",
        grasp_data.get_SE3_grasp_pose_in_palm_frame().shape,
    )
    print(
        "Contact points in CAD frame:",
        grasp_data.get_contact_points_in_cad_frame().shape,
    )
    print(
        "Contact normals in CAD frame:",
        grasp_data.get_contact_normals_in_cad_frame().shape,
    )
    print(
        "Link indices of contact points:",
        grasp_data.get_link_idxs_of_contact_points().shape,
    )
