import os, sys
import cv2
import numpy as np

from utils.lie.se3 import invSE3

import copy

# Intrinsic camera calibration option
NO_DISTORTION = (
    cv2.CALIB_FIX_K1
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
    | cv2.CALIB_FIX_K4
    | cv2.CALIB_FIX_K5
    | cv2.CALIB_FIX_K6
    | cv2.CALIB_ZERO_TANGENT_DIST
)


def get_object_points(checkerboard_dims=(7, 4), square_size=0.02725):
    """
    Generate object points for a checkerboard pattern.

    Args:
        checkerboard_dims (tuple): Dimensions of the checkerboard (rows, cols).
        square_size (float): Size of each square in meters.

    Returns:
        np.ndarray: Object points in 3D space (described from the Checkerboard frame).
    """
    objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : checkerboard_dims[0], 0 : checkerboard_dims[1]
    ].T.reshape(-1, 2)
    objp *= square_size
    return objp  # (checkerboard.size(), 3)


def load_image_and_pose_data(path):
    """
    Load image and pose data from a file.

    Args:
        path (str): Path to the file containing RGB image and SE(3) pose data.

    Returns:
        list: List of RGB images.
        list: List of SE(3) pose data.
    """
    list_time_idxs = []
    dict_of_images = {}
    dict_of_poses = {}
    for file in os.listdir(path):
        time_idx = file.split("_")[-1].split(".")[0]
        list_time_idxs.append(time_idx)
        if file.startswith("rgb_image"):
            img = cv2.imread(os.path.join(path, file))
            dict_of_images[time_idx] = img
        if file.startswith("link_pose_data"):
            pose = np.load(os.path.join(path, file))
            dict_of_poses[time_idx] = pose

    # unique time indices
    list_time_idxs = list(set(list_time_idxs))

    list_of_images = []
    list_of_link_poses = []
    for tidx in list_time_idxs:
        list_of_images.append(dict_of_images[tidx])
        list_of_link_poses.append(dict_of_poses[tidx])
    return list_of_images, list_of_link_poses


def corner_detection(
    list_of_images,
    list_of_poses,
    checkerboard_dims=(7, 4),
):
    """
    Detect corners in a list of images and refine their locations.
    Args:
        list_of_images (list): List of images to process.
        list_of_poses (list): List of SE(3) poses corresponding to the images.
        checkerboard_dims (tuple): Dimensions of the checkerboard (rows, cols).
    Returns:
        list: List of images with detected corners.
        list: List of SE(3) poses corresponding to the images.
        list: List of refined corner locations.
    """

    new_list_of_images = []
    new_list_of_poses = []
    list_of_image_points = []

    for i, img in enumerate(list_of_images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboard_dims,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            # Refine corner locations
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                0.00001,
            )
            corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            # image_points: 2D pixel locations of checkerboard corners
            new_list_of_images.append(list_of_images[i])
            new_list_of_poses.append(list_of_poses[i])
            list_of_image_points.append(corners_subpix)
        else:
            print(f"[Calibrator] Chessboard corners not found in {i+1}-th image")
    return new_list_of_images, new_list_of_poses, list_of_image_points


def intrinsic_calibration(
    list_of_image_points,
    img_shape=(640, 480),
    intr_cam_cal_option=NO_DISTORTION,
    checkerboard_dims=(7, 4),
    square_size=0.02725,
    init_camera_matrix=None,
    init_dist_coeffs=None,
):
    """
    Perform intrinsic camera calibration using detected corners.
    Args:
        list_of_image_points (list): List of detected corners in the images.
        img_shape (tuple): Shape of the images (width, height).
        intr_cam_cal_option: Calibration option. Different flags that may be zero or a combination of the following values:
            CALIB_USE_INTRINSIC_GUESS
                cameraMatrix contains valid initial values of fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image center ( imageSize is used), and focal distances are computed in a least-squares fashion. Note, that if intrinsic parameters are known, there is no need to use this function just to estimate extrinsic parameters. Use solvePnP instead.
            CALIB_FIX_PRINCIPAL_POINT
                The principal point is not changed during the global optimization. It stays at the center or at a different location specified when CALIB_USE_INTRINSIC_GUESS is set too.
            CALIB_FIX_ASPECT_RATIO
                The functions consider only fy as a free parameter. The ratio fx/fy stays the same as in the input cameraMatrix . When CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are ignored, only their ratio is computed and used further.
            CALIB_ZERO_TANGENT_DIST
                Tangential distortion coefficients are set to zeros and stay zero.
            CALIB_FIX_FOCAL_LENGTH
                The focal length is not changed during the global optimization if CALIB_USE_INTRINSIC_GUESS is set.
            CALIB_FIX_K1,..., CALIB_FIX_K6
                The corresponding radial distortion coefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
            CALIB_RATIONAL_MODEL
                Coefficients k4, k5, and k6 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients or more.
            CALIB_THIN_PRISM_MODEL
                Coefficients s1, s2, s3 and s4 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the thin prism model and return 12 coefficients or more.
            CALIB_FIX_S1_S2_S3_S4
                The thin prism distortion coefficients are not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
            CALIB_TILTED_MODEL
                Coefficients tauX and tauY are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the tilted sensor model and return 14 coefficients.
            CALIB_FIX_TAUX_TAUY
                The coefficients of the tilted sensor model are not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
        checkerboard_dims (tuple): Dimensions of the checkerboard (rows, cols).
        square_size (float): Size of each square in meters.
    Returns:
        np.ndarray: Camera matrix.
        np.ndarray: Distortion coefficients.
        rvecs, tvecs: Rotation and translation vectors (checkerboard to cam)
    """
    objp = get_object_points(
        checkerboard_dims=checkerboard_dims, square_size=square_size
    )

    list_obj_points = [objp] * len(list_of_image_points)

    ret, camera_matrix, dist_coeffs, rvecs_checker2cam, tvecs_checker2cam = (
        cv2.calibrateCamera(
            objectPoints=list_obj_points,
            imagePoints=list_of_image_points,
            imageSize=img_shape,
            cameraMatrix=init_camera_matrix,
            distCoeffs=init_dist_coeffs,
            flags=intr_cam_cal_option,
        )
    )

    if not ret:
        raise ValueError("[Calibrator] Intrinsic calibration failed. Not converged.")
    else:
        print(f"[Calibrator] Cal. intrinsic: \n {camera_matrix}")
        print(f"[Calibrator] Distortion coefficients: \n {dist_coeffs}")
    return camera_matrix, dist_coeffs, rvecs_checker2cam, tvecs_checker2cam


def compute_intr_cal_pixel_err(
    list_of_images,
    list_of_image_points,
    rvecs_checker2cam,
    tvecs_checker2cam,
    camera_matrix,
    dist_coeffs,
    checkerboard_dims=(7, 4),
    square_size=0.02725,
    visualize=False,
):
    objp = get_object_points(
        checkerboard_dims=checkerboard_dims, square_size=square_size
    )
    list_object_points = [objp] * len(list_of_image_points)

    total_error = 0
    total_points = 0

    list_of_errors = []
    for i, img in enumerate(list_object_points):
        img = list_of_images[i].copy()  # BGR image
        img_points = list_of_image_points[i]
        obj_points = list_object_points[i]

        # Project 3D points to image plane
        imgpoints2, _ = cv2.projectPoints(
            obj_points,
            rvecs_checker2cam[i],
            tvecs_checker2cam[i],
            camera_matrix,
            dist_coeffs,
        )

        # Compute per-image error
        error = np.linalg.norm(img_points - imgpoints2, axis=-1).mean()
        total_error += error * len(obj_points)
        total_points += len(obj_points)

        list_of_errors.append(error)

        # Display the image with reprojection
        if visualize:
            # Draw original detected corners (green) and reprojected (red)
            for p1, p2 in zip(img_points, imgpoints2):
                cv2.circle(
                    img, tuple(p1.ravel().astype(int)), 4, (0, 255, 0), -1
                )  # detected
                cv2.circle(
                    img, tuple(p2.ravel().astype(int)), 4, (0, 0, 255), -1
                )  # reprojected

            window_name = f"Reprojection {i}"
            cv2.imshow(window_name, img)
            print(f"[{i}] Reprojection error: {error:.4f}")

            cv2.waitKey(500)
            while True:
                key = cv2.waitKey(100)  # Check every 100 ms
                try:
                    # Try to get window property, if it returns < 0, it's closed
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break  # Window was already destroyed
            cv2.destroyWindow(window_name)

    # Overall reprojection error
    mean_error = total_error / total_points
    return mean_error, list_of_errors


def filter_bad_data_with_intr_cal_results(
    list_of_images,
    list_of_poses,
    list_of_image_points,
    rvecs_checker2cam,
    tvecs_checker2cam,
    camera_matrix,
    dist_coeffs,
    checkerboard_dims=(7, 4),
    square_size=0.02725,
    err_thr=0.5,
):
    mean_err, list_of_errs = compute_intr_cal_pixel_err(
        list_of_images,
        list_of_image_points,
        rvecs_checker2cam,
        tvecs_checker2cam,
        camera_matrix,
        dist_coeffs,
        checkerboard_dims=checkerboard_dims,
        square_size=square_size,
        visualize=False,
    )
    new_list_of_images = []
    new_list_of_poses = []
    new_list_of_image_points = []

    for i, (img, pose, img_pts, err) in enumerate(
        zip(list_of_images, list_of_poses, list_of_image_points, list_of_errs)
    ):
        if err < err_thr:
            new_list_of_images.append(img)
            new_list_of_poses.append(pose)
            new_list_of_image_points.append(img_pts)
        else:
            print(f"[Calibrator] {i+1}-th image is filtered out with error: {err:.4f}")

    return new_list_of_images, new_list_of_poses, new_list_of_image_points


def extrinsic_calibration(
    rvecs_checker2cam,
    tvecs_checker2cam,
    list_of_poses_link2base,
):
    """
    Perform extrinsic calibration using the hand-eye calibration method.
    Args:
        rvecs_checker2cam (list): List of rotation vectors (checkerboard to cam).
        tvecs_checker2cam (list): List of translation vectors (checkerboard to cam).
        list_of_poses_link2base (list): List of SE(3) poses (link to base).
    Returns:
        np.ndarray: Transformation matrix from base to camera.
        np.ndarray: Transformation matrix from link to checkerboard.
    """
    R_cam2checker = []
    t_cam2checker = []
    for rvec, tvec in zip(rvecs_checker2cam, tvecs_checker2cam):
        rot = cv2.Rodrigues(rvec)[0]
        trans = tvec
        R_cam2checker.append(copy.deepcopy(rot.T))
        t_cam2checker.append(copy.deepcopy(-rot.T @ trans))

    R_base2link = []
    t_base2link = []
    for pose in list_of_poses_link2base:
        rot = pose[:3, :3]
        trans = pose[:3, 3:4]
        R_base2link.append(copy.deepcopy(rot.T))
        t_base2link.append(copy.deepcopy(-rot.T @ trans))

    R_base2cam, t_base2cam, R_link2checker, t_link2checker = (
        cv2.calibrateRobotWorldHandEye(
            R_cam2checker,
            t_cam2checker,
            R_base2link,
            t_base2link,
        )
    )

    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R_base2cam
    T_base2cam[:3, 3:4] = t_base2cam
    print(f"[Calibrator] Base to camera transformation: \n {T_base2cam}")

    T_link2checker = np.eye(4)
    T_link2checker[:3, :3] = R_link2checker
    T_link2checker[:3, 3:4] = t_link2checker
    print(f"[Calibrator] Link to checkerboard transformation: \n {T_link2checker}")
    return T_base2cam, T_link2checker


def compute_cal_pixel_err(
    T_link2checker,
    T_base2cam,
    camera_matrix,
    dist_coeffs,
    list_of_images,
    list_of_poses,
    list_of_image_points,
    checkerboard_dims=(7, 4),
    square_size=0.02725,
):
    objp = get_object_points(
        checkerboard_dims=checkerboard_dims, square_size=square_size
    )
    list_object_points = [objp] * len(list_of_image_points)

    list_errors = []

    for _, T_link2base, obj_points, img_points in zip(
        list_of_images,
        list_of_poses,
        list_object_points,
        list_of_image_points,
    ):
        T_cam2checker = T_link2checker @ invSE3(T_base2cam @ T_link2base)
        T_checker2cam = invSE3(T_cam2checker)

        R_checker2cam = T_checker2cam[:3, :3]
        tvec_checker2cam = T_checker2cam[:3, 3]
        rvec_checker2cam, _ = cv2.Rodrigues(R_checker2cam)

        projected, _ = cv2.projectPoints(
            obj_points, rvec_checker2cam, tvec_checker2cam, camera_matrix, dist_coeffs
        )
        list_errors.append(np.linalg.norm(projected - img_points, axis=-1).mean())
        mean_error = np.mean(list_errors)
    return mean_error, list_errors
