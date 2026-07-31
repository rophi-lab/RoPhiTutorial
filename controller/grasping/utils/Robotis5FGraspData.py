import numpy as np
import os

from controller.grasping.utils.GraspData import (
    GraspData,
)

ROBOTIS_5F_HAND_LINK_NAMES = [
    "palm",
    "finger_r_link_1_thumb1",
    "finger_r_link_1_thumb2",
    "finger_r_link_1_thumb3",
    "finger_r_link_1_thumb4",
    "finger_r_link_2_index1",
    "finger_r_link_2_index2",
    "finger_r_link_2_index3",
    "finger_r_link_2_index4",
    "finger_r_link_3_middle1",
    "finger_r_link_3_middle2",
    "finger_r_link_3_middle3",
    "finger_r_link_3_middle4",
    "finger_r_link_4_ring1",
    "finger_r_link_4_ring2",
    "finger_r_link_4_ring3",
    "finger_r_link_4_ring4",
    "finger_r_link_5_little1",
    "finger_r_link_5_little2",
    "finger_r_link_5_little3",
    "finger_r_link_5_little4",
]

CANDIDATE_CONTACT_PATH = "assets/grasp_templates/robotis_5f/candidate_contact_points"


class Robotis5FGraspData(GraspData):
    def __init__(self, grasp_predictions_path):
        super().__init__(grasp_predictions_path)

        self._mapper_idx_to_link_name = {
            i: link_name for i, link_name in enumerate(ROBOTIS_5F_HAND_LINK_NAMES)
        }
        self._set_link_names_of_contact_points()
        self._set_candidate_contact_info(CANDIDATE_CONTACT_PATH)

    def get_candidate_contact_points_in_link_frame(self):
        return self._candidate_contact_points_in_link_frame

    def get_candidate_contact_normals_in_link_frame(self):
        return self._candidate_contact_normals_in_link_frame

    def get_candidate_contact_link_indices(self):
        return self._candidate_contact_link_indices

    def get_candidate_contact_link_names(self):
        return self._candidate_contact_link_names

    def _set_link_names_of_contact_points(self):
        num_grasps = self._link_idxs_of_contact_points.shape[0]
        P_max = self._link_idxs_of_contact_points.shape[1]
        self._link_names_of_contact_points = np.empty(
            (num_grasps, P_max), dtype=object
        )
        for i in range(num_grasps):
            P_i = self._num_contacts_per_grasp[i]
            for j in range(P_i):
                link_idx = self._link_idxs_of_contact_points[i][j]
                self._link_names_of_contact_points[i][j] = (
                    self._mapper_idx_to_link_name[link_idx]
                )

    def _set_candidate_contact_info(self, candidate_contact_path):
        # M : number of candidate contact points
        self._candidate_contact_points_in_link_frame = np.load(
            os.path.join(
                candidate_contact_path, "candidate_contact_points_in_link_frame.npy"
            )
        )  # (M, 3)
        self._candidate_contact_normals_in_link_frame = np.load(
            os.path.join(
                candidate_contact_path, "candidate_contact_normals_in_link_frame.npy"
            )
        )  # (M, 3)
        self._candidate_contact_link_indices = np.load(
            os.path.join(candidate_contact_path, "candidate_contact_link_indices.npy")
        )  # (M,)
        self._candidate_contact_link_names = [
            self._mapper_idx_to_link_name[link_idx]
            for link_idx in self._candidate_contact_link_indices
        ]  # (M,)
