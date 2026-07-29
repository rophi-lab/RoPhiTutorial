import sys

sys.path.insert(0, "/opt/openrobots/lib/python3.10/site-packages")
from pinocchio import casadi as cpin
import casadi as ca
import pinocchio as pin

from pinocchio import FrameType, ReferenceFrame
import numpy as np

import copy


def get_fk_link_poses(q, cpin_model, cpin_data, link_names):
    """
    Get the poses of multiple links in the world frame.
    @param[in] q: Joint configuration
    @param[in] cpin_model: Pinocchio model
    @param[in] cpin_data: Pinocchio data
    @param[in] link_names: List of link names
    @return: List of poses of the links in the world frame as 4x4 matrices
    """
    cpin.forwardKinematics(cpin_model, cpin_data, q)
    cpin.updateFramePlacements(cpin_model, cpin_data)
    link_poses = []
    for link_name in link_names:
        for frame_id, frame in enumerate(cpin_model.frames):
            if frame.type == FrameType.BODY:
                if frame.name == link_name:
                    ee_frame_id = frame_id
                    break
        link_pose = cpin_data.oMf[ee_frame_id]
        link_pose_pos = link_pose.translation
        link_pose_rot = link_pose.rotation

        upper = ca.horzcat(link_pose_rot, link_pose_pos)
        lower = ca.horzcat(ca.SX.zeros(1, 3), ca.SX(1))
        link_pose_mat = ca.vertcat(upper, lower)
        link_poses.append(copy.deepcopy(link_pose_mat))
    return copy.deepcopy(link_poses)


def get_fk_link_pose(q, cpin_model, cpin_data, link_name):
    """
    Get the pose of a link in the world frame.
    @param[in] q: Joint configuration
    @param[in] cpin_model: Pinocchio model
    @param[in] cpin_data: Pinocchio data
    @param[in] link_name: Name of the link
    @return: Pose of the link in the world frame as a 4x4 matrix
    """
    cpin.forwardKinematics(cpin_model, cpin_data, q)
    cpin.updateFramePlacements(cpin_model, cpin_data)
    for frame_id, frame in enumerate(cpin_model.frames):
        if frame.type == FrameType.BODY:
            if frame.name == link_name:
                ee_frame_id = frame_id
                break
    link_pose = cpin_data.oMf[ee_frame_id]
    link_pose_pos = link_pose.translation
    link_pose_rot = link_pose.rotation

    upper = ca.horzcat(link_pose_rot, link_pose_pos)
    lower = ca.horzcat(ca.SX.zeros(1, 3), ca.SX(1))
    link_pose_mat = ca.vertcat(upper, lower)
    return copy.deepcopy(link_pose_mat)


def get_link_Jacobian(q, cpin_model, cpin_data, link_name):
    """
    Get the Jacobian of a link in the world frame.
    @param[in] q: Joint configuration
    @param[in] cpin_model: Pinocchio model
    @param[in] cpin_data: Pinocchio data
    @param[in] link_name: Name of the link
    @return: Jacobian of the link in the world frame as a 6xN matrix
    """
    cpin.forwardKinematics(cpin_model, cpin_data, q)
    cpin.updateFramePlacements(cpin_model, cpin_data)
    for frame_id, frame in enumerate(cpin_model.frames):
        if frame.type == FrameType.BODY:
            if frame.name == link_name:
                ee_frame_id = frame_id
                break
    # NOTE J = (Jp, JR)
    J = cpin.getFrameJacobian(
        cpin_model,
        cpin_data,
        ee_frame_id,
        ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    # Change it to (JR, Jp)
    J = ca.vertcat(J[3:6, :], J[0:3, :])
    return copy.deepcopy(J)
