import numpy as np

import mujoco as mj

import matplotlib.pyplot as plt

from utils.perception.camera import pixel_to_camera_with_K


def initialize_camera_renderer(
    mj_model: mj.MjModel,
    height: int = 480,
    width: int = 640,
):
    """
    Initialize a camera renderer for the given camera name.
    In theory we don't need multiple renderers. However, if we
    want to render cameras with different resolutions, we need to
    create multiple renderers.

    @param[in]: mj_model (mj.MjModel): The mujoco model object.
    @param[in]: height (int): The height of the camera image.
    @param[in]: width (int): The width of the camera image.

    @Returns:
        mj.Renderer: The initialized camera renderer.
    """

    renderer = mj.Renderer(mj_model, height, width)

    return renderer


def render_image_from_camera(
    renderer: mj.Renderer,
    mj_data: mj.MjData,
    camera_name: str,
    rgb_enable: bool = True,
    depth_enable: bool = True,
):
    """
    Render an image from the specified camera.


    @param[in]: renderer (mj.Renderer): The mujoco camera renderer.
    @param[in]: mj_data (mj.MjData): The mujoco data object.
    @param[in]: camera_name (str): The name of the camera to render.
    @param[in]: rgb_enable (bool): Whether to enable RGB rendering.
    @param[in]: depth_enable (bool): Whether to enable depth rendering.
    @param[in]: enable_depth (bool): Whether to enable depth rendering.

    @Returns:
        (rgb_image, depth_image)
        tuple(np.ndarray, np.ndarray)
        (h,w,3), (h,w)

        if rgb is not enabled, rgb_image is None
        if depth is not enabled, depth_image is None
    """
    renderer.update_scene(mj_data, camera=camera_name)
    rgb_image = None
    depth_image = None
    if rgb_enable:
        # Get the RGB image (height, width, 3)
        rgb_image = renderer.render()

    if depth_enable:
        # Get the depth image (height, width)
        renderer.enable_depth_rendering()
        depth_image = renderer.render()
        renderer.disable_depth_rendering()
        # In mujoco, camera is viewing -z direction,
        # thus z is negative depth.
        depth_image = depth_image

    # return np.flipud(rgb_image).copy(), np.flipud(depth_image).copy()
    return rgb_image, depth_image


def get_camera_intrinsics(model, camera_name, height, width):
    cam_id = model.camera(name=camera_name).id
    fovy_deg = model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy_deg)

    # Vertical focal length
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy  # in mujoco camera frame, x is pointing to the right

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def get_camera_extrinsics(model, data, camera_name):
    cam_id = model.camera(name=camera_name).id

    # Camera pose in world frame (position and rotation matrix)
    # cam_pos = model.cam_pos[cam_id]
    # cam_mat = model.cam_mat[cam_id].reshape(3, 3)  # column-major rotation matrix
    extrinsics = np.eye(4)
    # 3x3 rotation matrix (world to camera orientation)
    R_world = data.cam_xmat[cam_id].reshape(3, 3)
    # print("R_world: ", R_world)
    # position of the camera in world frame
    p_world = data.cam_xpos[cam_id]  # shape (3,)
    # print("p_world: ", p_world)
    # In MuJoCo, cam_mat gives camera orientation columns in world frame
    # So the camera-to-world transform is [R | t], world-to-camera is R.T and -R.T @ t

    # World to camera rotation and translation
    R_world_to_cam = (R_world).T
    # R_world_to_cam = (R_world).T
    p_world_to_cam = -R_world_to_cam @ p_world

    T_camzforward_cam = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    extrinsics[:3, :] = np.hstack([R_world_to_cam, p_world_to_cam.reshape(3, 1)])  # 4x4

    return T_camzforward_cam @ extrinsics
