# NOTE: try to import casadi version of pinocchio first
try:
    sys.path.insert(0, "/opt/openrobots/lib/python3.10/site-packages")
    from pinocchio import casadi as cpin
    import casadi as ca
    import pinocchio as pin
except:
    import pinocchio as pin

from pinocchio import ReferenceFrame
import numpy as np

import copy


def _frame_id(pin_model, frame_name: str) -> int:
    """Resolve a Pinocchio frame by name (BODY, OP_FRAME, FIXED_JOINT, …)."""
    if not pin_model.existFrame(frame_name):
        raise ValueError(f"Frame {frame_name!r} not found in Pinocchio model")
    return pin_model.getFrameId(frame_name)


def get_fk_link_poses(q, pin_model, pin_data, link_names):
    """
    Get the poses of multiple links in the world frame.
    @param[in] q: Joint configuration
    @param[in] pin_model: Pinocchio model
    @param[in] pin_data: Pinocchio data
    @param[in] link_names: List of link names
    @return: List of poses of the links in the world frame as 4x4 matrices
    """
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    link_poses = []
    for link_name in link_names:
        ee_frame_id = _frame_id(pin_model, link_name)
        link_pose = pin_data.oMf[ee_frame_id]
        link_pose_pos = link_pose.translation
        link_pose_rot = link_pose.rotation
        link_pose_mat = np.eye(4)
        link_pose_mat[:3, :3] = link_pose_rot
        link_pose_mat[:3, 3] = link_pose_pos
        link_poses.append(copy.deepcopy(link_pose_mat))
    return copy.deepcopy(link_poses)


def get_fk_link_pose(q, pin_model, pin_data, link_name):
    """
    Get the pose of a link in the world frame.
    @param[in] q: Joint configuration
    @param[in] pin_model: Pinocchio model
    @param[in] pin_data: Pinocchio data
    @param[in] link_name: Name of the link
    @return: Pose of the link in the world frame as a 4x4 matrix
    """
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    ee_frame_id = _frame_id(pin_model, link_name)
    link_pose = pin_data.oMf[ee_frame_id]
    link_pose_pos = link_pose.translation
    link_pose_rot = link_pose.rotation
    link_pose_mat = np.eye(4)
    link_pose_mat[:3, :3] = link_pose_rot
    link_pose_mat[:3, 3] = link_pose_pos
    return copy.deepcopy(link_pose_mat)


def get_link_Jacobian(q, pin_model, pin_data, link_name):
    """
    Get the Jacobian of a link in the world frame.

    Returns a 6xn matrix stacked as (JR; Jp) in LOCAL_WORLD_ALIGNED:
    angular (world) then linear (world) of the frame origin.

    Uses ``computeFrameJacobian`` (FK + frame update + Jacobian in one call).
    """
    ee_frame_id = _frame_id(pin_model, link_name)
    # Pinocchio native layout is (Jp; JR); we return (JR; Jp) for this codebase.
    J_pin = pin.computeFrameJacobian(
        pin_model,
        pin_data,
        q,
        ee_frame_id,
        ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    J = np.vstack((J_pin[3:, :], J_pin[:3, :]))
    return copy.deepcopy(J)


def get_link_Jacobians(q, pin_model, pin_data, link_names):
    """
    Get the Jacobians of multiple links in the world frame.
    @param[in] q: Joint configuration
    @param[in] pin_model: Pinocchio model
    @param[in] pin_data: Pinocchio data
    @param[in] link_names: List of link names
    @return: List of Jacobians of the links in the world frame as 6xN matrices
    """
    J_list = []
    for link_name in link_names:
        J_list.append(get_link_Jacobian(q, pin_model, pin_data, link_name))
    return copy.deepcopy(J_list)


def get_link_velocity(q, qd, pin_model, pin_data, link_name):
    """
    Get the linear and angular velocities of a link in the world frame.

    @param[in] q: Joint configuration
    @param[in] qd: Joint velocity
    @param[in] pin_model: Pinocchio model
    @param[in] pin_data: Pinocchio data
    @param[in] link_name: Name of the link
    @return: Linear and angular velocities of the link in the world frame
    """
    pin.forwardKinematics(pin_model, pin_data, q, qd)
    pin.updateFramePlacements(pin_model, pin_data)
    ee_frame_id = _frame_id(pin_model, link_name)
    v = pin.getFrameVelocity(
        pin_model,
        pin_data,
        ee_frame_id,
        ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    return copy.deepcopy(v.linear), copy.deepcopy(v.angular)
