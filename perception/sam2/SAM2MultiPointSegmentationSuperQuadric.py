import os
import numpy as np
import time
import cv2

from typing import List, Tuple

import torch

import open3d as o3d

from data_type.basic_types.Point3DData import Point3DData
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.NamedVecListData import NamedVecListData
from perception.BasePerception import BasePerception
from utils.perception.camera import convert_pixel_to_world
from utils.lie.se3 import invSE3
from utils.visualization.bounding_box import draw_posed_3d_box, draw_xyz_axis
from utils.shape_primitives.functions import bb2superellipsoids


class SAM2MultiPointSegmentationSuperQuadric(BasePerception):
    def __init__(self, config):
        super().__init__(config)
        self.name = "SAM2MultiPointSegmentationSuperQuadric"

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration parameters
        self._update_freq = config.get("update_freq", 10.0)
        self._update_dt = 1 / self._update_freq

        # SAM2 configuration
        self._sam2_ckpt = config.get("sam2_ckpt", "sam2.1_hiera_large.pt")
        self._sam2_config_path = config.get(
            "sam2_config_path", "configs/perception/sam2/sam2.1_hiera_l.yaml"
        )

        # Point cloud processing
        self._pcd_stat_outlier_removal = config.get("pcd_stat_outlier_removal", {}).get(
            "enable", True
        )
        self._pcd_stat_outlier_removal_nb_neighbors = config.get(
            "pcd_stat_outlier_removal", {}
        ).get("nb_neighbors", 20)
        self._pcd_stat_outlier_removal_std_ratio = config.get(
            "pcd_stat_outlier_removal", {}
        ).get("std_ratio", 1.5)

        self._pcd_radius_outlier_removal = config.get(
            "pcd_radius_outlier_removal", {}
        ).get("enable", True)
        self._pcd_radius_outlier_removal_radius = config.get(
            "pcd_radius_outlier_removal", {}
        ).get("radius", 0.03)

        # Bounding box fitting
        self._bbox_method = config.get("bbox_method", "oriented")

        self._publish_fixed_table = config.get("publish_fixed_table", False)

        # Visualization options
        self._visualize_masks = config.get("visualize_masks", True)
        self._visualize_bboxes = config.get("visualize_bboxes", True)
        self._visualize_every_point_cloud = config.get(
            "visualize_every_point_cloud", False
        )
        self._visualize_all_point_clouds = config.get(
            "visualize_all_point_clouds", False
        )
        self._visualize_pcd_outliers = config.get("visualize_pcd_outliers", False)
        self._save_results = config.get("save_results", False)
        self._save_dir = config.get("save_dir", "debug_output/sam2_segmentation")

        # Channels
        self._rgbd_channel = config["sub_manager"]["rgbd_channel"]
        self._superellipsoid_pub_channel = config["pub_manager"].get(
            "named_vec_list_channel", "superellipsoids"
        )

        # Camera parameters
        self._cam_intrinsic = np.eye(3)
        self._cam2world = np.eye(4)
        self._depth_factor = 1.0
        self._img_height = 480
        self._img_width = 640

        # State variables
        self._clicked_points: List[Tuple[int, int]] = []
        self._last_update_t = time.time()
        self._window_name = "SAM2 Multi-Point Segmentation"
        self._segmentation_complete = False
        self._masks = []
        self._point_clouds = []
        self._oriented_bboxes = []

        # SAM2 model
        self._sam2_predictor = None
        self._inference_state = None

        self._segmentation_complete = False

        # Create output directory if saving results
        if self._save_results:
            os.makedirs(self._save_dir, exist_ok=True)
            os.makedirs(os.path.join(self._save_dir, "masks"), exist_ok=True)
            os.makedirs(os.path.join(self._save_dir, "point_clouds"), exist_ok=True)
            os.makedirs(os.path.join(self._save_dir, "bboxes"), exist_ok=True)

    def initialize(self):
        """Initialize the perception module"""

        # print("Initializing SAM2MultiPointSegmentation...")
        # Initialize GUI AFTER SAM2 initialization
        self._init_gui()

        # Wait for camera parameters
        cam_param_initialized = False
        timeout_start = time.time()
        timeout_duration = 10.0  # 10 seconds timeout

        while not cam_param_initialized:
            if time.time() - timeout_start > timeout_duration:
                print(
                    "Warning: Camera parameter initialization timed out. Using default values."
                )
                break

            if not self.extr_sub_que_dict[self._rgbd_channel + "_info"].empty():
                cam_info = self.extr_sub_que_dict[self._rgbd_channel + "_info"].get()
                if isinstance(cam_info, CameraInfoData):
                    self._cam_intrinsic = cam_info.get_intrinsic()
                    self._cam2world = invSE3(cam_info.get_extrinsic())
                    self._depth_factor = cam_info.get_depth_factor()
                    self._img_height, self._img_width = cam_info.get_img_dim()
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")
            else:
                time.sleep(0.1)  # Small delay to prevent busy waiting

            # Initialize SAM2 if available
            # if SAM2_AVAILABLE:
        self._init_sam2()
        # else:
        # print("Warning: SAM2 not available. Segmentation will not work.")

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    def _init_sam2(self):
        """Initialize SAM2 model"""
        try:
            print("Initializing SAM2 model...")
            # we need to include sam2 after initializing the opencv window
            # otherwise, the opencv window will be blocked
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._sam2_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-large"
            )
            print("SAM2 model initialized")
        except Exception as e:
            print(f"Error initializing SAM2: {e}")
            self._sam2_predictor = None

    def _process(self):
        """Main processing loop"""

        if not self.extr_sub_que_dict[self._rgbd_channel].empty():

            # Get latest RGBD data
            while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

            rgb_image = rgbd_data.get_rgb_image()
            depth_image = rgbd_data.get_depth_image()

            if time.time() - self._last_update_t < self._update_dt:
                return
            else:
                self._last_update_t = time.time()

            # Convert to BGR for OpenCV display
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # Draw clicked points
            for i, point in enumerate(self._clicked_points):
                color = (
                    (0, 255, 0) if i < len(self._clicked_points) - 1 else (0, 0, 255)
                )
                cv2.circle(bgr_image, point, radius=5, color=color, thickness=-1)
                cv2.putText(
                    bgr_image,
                    str(i + 1),
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            cv2.imshow(self._window_name, bgr_image)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if (
                key == ord("s")
                and len(self._clicked_points) > 0
                and not self._segmentation_complete
            ):
                self._perform_segmentation(rgb_image, depth_image, rgbd_data)
            elif key == ord("r"):
                self._reset_segmentation()
            elif key == ord("q"):
                print("Quitting...")
                return

            if self._segmentation_complete:
                self._publish_results(rgbd_data)

            # Handle new clicked points
            if self.clicked_point is not None:
                self._clicked_points.append(self.clicked_point)
                self.clicked_point = None

    def _perform_segmentation(self, rgb_image, depth_image, rgbd_data):
        """Perform SAM2 segmentation for all clicked points"""
        if self._sam2_predictor is None:
            print("SAM2 not available for segmentation")
            return

        print(f"Starting segmentation for {len(self._clicked_points)} points...")

        # Initialize SAM2 inference state
        try:

            # start counting time
            start_time = time.time()
            self._sam2_predictor.set_image(rgb_image)

            # Process each point
            self._masks = []
            self._point_clouds = []
            self._oriented_bboxes = []

            masks, scores, logits = self._sam2_predictor.predict(
                np.array(self._clicked_points).reshape(-1, 1, 2),
                np.ones(len(self._clicked_points)).reshape(-1, 1),
                multimask_output=False,
            )
            # end counting time
            end_time = time.time()
            # print(f"SAM2 inference time: {end_time - start_time} seconds")

            for i, clicked_point in enumerate(self._clicked_points):
                print(
                    f"Processing point {i+1}/{len(self._clicked_points)}: {clicked_point}"
                )

                # Get mask for this point
                mask_logits = masks[i]  # Get mask for object i+1

                # Convert logits to probability mask
                if hasattr(mask_logits, "shape") and len(mask_logits.shape) > 0:
                    mask_prob = 1.0 / (1.0 + np.exp(-mask_logits))
                    mask = mask_prob > 0.5
                else:
                    mask = np.zeros((self._img_height, self._img_width), dtype=bool)

                self._masks.append(mask)

                # Extract point cloud from masked depth
                point_cloud = self._extract_point_cloud_from_mask(
                    mask, depth_image, rgb_image, clicked_point
                )
                self._point_clouds.append(point_cloud)

                # Fit oriented bounding box
                if point_cloud is not None and len(point_cloud.points) > 10:
                    bbox = self._fit_oriented_bounding_box(point_cloud)
                    self._oriented_bboxes.append(bbox)
                else:
                    self._oriented_bboxes.append(None)

                # visualize point cloud
                if self._visualize_every_point_cloud:
                    self._visualize_point_cloud(
                        point_cloud, i, self._oriented_bboxes[i], rgb_image
                    )

                # Save results if enabled
                if self._save_results:
                    self._save_segmentation_results(
                        i, mask, point_cloud, self._oriented_bboxes[i]
                    )

            # Visualize all results together
            if self._visualize_masks and len(self._masks) > 0:
                self._visualize_all_masks(rgb_image, self._masks, self._clicked_points)

            if self._visualize_bboxes and len(self._oriented_bboxes) > 0:
                self._visualize_all_bboxes(
                    rgb_image, self._oriented_bboxes, self._clicked_points
                )

            if self._visualize_all_point_clouds and len(self._point_clouds) > 0:
                self._visualize_all_point_clouds()

            # Publish results
            # self._publish_results(rgbd_data)

            self._segmentation_complete = True
            print("Segmentation completed successfully!")

        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback

            traceback.print_exc()

    def _extract_point_cloud_from_mask(
        self, mask, depth_image, rgb_image=None, clicked_point=None
    ):
        """Extract point cloud from masked depth image"""
        # Find valid depth pixels within the mask
        valid_mask = mask & (depth_image > 0)
        if valid_mask.ndim == 3:
            valid_mask = np.squeeze(valid_mask, axis=0)
        y_coords, x_coords = np.where(valid_mask)
        if len(y_coords) == 0:
            return None

        # Extract depth values
        depths = depth_image[y_coords, x_coords]

        # Convert to camera coordinates
        fx = self._cam_intrinsic[0, 0]
        fy = self._cam_intrinsic[1, 1]
        cx = self._cam_intrinsic[0, 2]
        cy = self._cam_intrinsic[1, 2]

        # Convert to camera coordinates
        z = depths / self._depth_factor
        x = (x_coords - cx) * z / fx
        y = (y_coords - cy) * z / fy

        # Stack coordinates
        points_cam = np.stack([x, y, z], axis=1)

        # Transform to world coordinates
        points_world = (self._cam2world[:3, :3] @ points_cam.T).T + self._cam2world[
            :3, 3
        ]

        # if len(pcd_clean.points) < 10:
        #     return None

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)

        # Add RGB colors if available
        if rgb_image is not None:
            # Extract RGB values for the masked pixels
            rgb_values = rgb_image[y_coords, x_coords]
            # Normalize RGB values to [0, 1] range
            rgb_normalized = rgb_values.astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb_normalized)

            # Remove outliers
        if self._pcd_stat_outlier_removal:
            _, ind_stat = pcd.remove_statistical_outlier(
                nb_neighbors=self._pcd_stat_outlier_removal_nb_neighbors,
                std_ratio=self._pcd_stat_outlier_removal_std_ratio,
            )

        if self._pcd_radius_outlier_removal:
            # Distance-based outlier removal if clicked point is provided
            if clicked_point is not None:

                clicked_point_world = convert_pixel_to_world(
                    clicked_point,
                    depth_image,
                    self._cam_intrinsic,
                    self._cam2world,
                    depth_factor=self._depth_factor,
                )
                # Calculate distances from clicked point to all points
                points_array = points_world
                distances = np.linalg.norm(points_array - clicked_point_world, axis=1)

                # Keep points within a reasonable distance (e.g., 0.2 meters)
                max_distance = self._pcd_radius_outlier_removal_radius
                close_indices = np.where(distances <= max_distance)[0]

            # Find the intersection of the two indices (points that pass both filters)
            # ind_stat contains indices from original pcd (statistical filtering)
            # close_indices contains indices from pcd_world_stat (distance filtering)
            # We need to map distance indices back to original point cloud indices
            if self._pcd_stat_outlier_removal:
                # Both filters applied: find intersection
                ind_union = np.intersect1d(ind_stat, close_indices)
            else:
                # Only distance filter applied
                ind_union = close_indices

            pcd_world_clean = pcd.select_by_index(ind_union)

            if self._visualize_pcd_outliers:
                # Visualize
                inlier_cloud = pcd.select_by_index(ind_union)
                inlier_cloud.paint_uniform_color([0, 1, 0])  # Green for inliers
                outlier_cloud = pcd.select_by_index(ind_union, invert=True)
                outlier_cloud.paint_uniform_color([1, 0, 0])  # Red for removed points

                o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        else:
            pcd_world_clean = pcd

        # Optional: estimate normals
        # if len(pcd_world_clean.points) > 10:
        #     pcd.estimate_normals(
        #         search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        #     )

        return pcd_world_clean

    def _fit_oriented_bounding_box(self, point_cloud):
        """Fit oriented bounding box to point cloud"""
        try:

            # Fit oriented bounding box
            if self._bbox_method == "oriented":
                bbox = point_cloud.get_oriented_bounding_box()
            elif self._bbox_method == "minimal":
                bbox = point_cloud.get_minimal_oriented_bounding_box()
            elif self._bbox_method == "axis_aligned":
                bbox = point_cloud.get_axis_aligned_bounding_box()
                bbox = bbox.get_oriented_bounding_box()

            return bbox

        except Exception as e:
            print(f"Error fitting bounding box: {e}")
            return None

    def _visualize_point_cloud(self, point_cloud, index, bbox=None, rgb_image=None):
        """Visualize point cloud and bounding box using Open3D"""
        try:
            if point_cloud is None or len(point_cloud.points) == 0:
                print(f"Point cloud {index+1} is empty or None")
                return

            # Get points as numpy array
            points = np.asarray(point_cloud.points)

            # Create a new point cloud for visualization
            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(points)

            # Define color palette for fallback and bounding box
            color_palette = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
                [0.5, 0.0, 1.0],  # Purple
            ]
            color = color_palette[index % len(color_palette)]

            # Set colors based on original RGB image if available
            if (
                rgb_image is not None
                and hasattr(point_cloud, "colors")
                and len(point_cloud.colors) > 0
            ):
                # Use original colors from the point cloud
                colors = np.asarray(point_cloud.colors)
                vis_pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Fallback to uniform color based on index
                vis_pcd.paint_uniform_color(color)

            # Create coordinate frame for reference
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )

            # Prepare geometries for visualization
            geometries = [vis_pcd, coordinate_frame]

            # Add bounding box if available
            if bbox is not None:
                bbox_mesh = self._create_bbox_mesh(bbox, color)
                if bbox_mesh is not None:
                    geometries.append(bbox_mesh)

            # Create visualization window
            window_name = f"Point Cloud {index+1}"

            # Visualize the point cloud and bounding box
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                width=800,
                height=600,
                point_show_normal=False,
                mesh_show_wireframe=False,
                mesh_show_back_face=False,
            )

            print(f"Visualized point cloud {index+1} with {len(points)} points")

        except Exception as e:
            print(f"Error visualizing point cloud {index+1}: {e}")
            import traceback

            traceback.print_exc()

    def _create_bbox_mesh(self, bbox, color):
        """Create a mesh representation of the bounding box"""
        try:

            # Set color for the wireframe (green)
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
            line_set.paint_uniform_color([0.0, 1.0, 0.0])  # Green color

            return line_set

        except Exception as e:
            print(f"Error creating bounding box mesh: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _visualize_all_point_clouds(self):
        """Visualize all point clouds together"""
        try:
            if not self._point_clouds or all(pcd is None for pcd in self._point_clouds):
                print("No point clouds to visualize")
                return

            # Create coordinate frame for reference
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )

            # Prepare all point clouds for visualization
            geometries = [coordinate_frame]
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
                [0.5, 0.0, 1.0],  # Purple
            ]

            for i, (pcd, bbox) in enumerate(
                zip(self._point_clouds, self._oriented_bboxes)
            ):
                if pcd is not None and len(pcd.points) > 0:
                    # Create a copy for visualization
                    vis_pcd = o3d.geometry.PointCloud()
                    vis_pcd.points = pcd.points

                    # Set colors based on original RGB if available
                    if hasattr(pcd, "colors") and len(pcd.colors) > 0:
                        # Use original colors from the point cloud
                        colors_array = np.asarray(pcd.colors)
                        vis_pcd.colors = o3d.utility.Vector3dVector(colors_array)
                    else:
                        # Fallback to uniform color based on index
                        color = colors[i % len(colors)]
                        vis_pcd.paint_uniform_color(color)

                    geometries.append(vis_pcd)

                    # Add bounding box if available
                    if bbox is not None:
                        bbox_mesh = self._create_bbox_mesh(
                            bbox, [0.0, 1.0, 0.0]
                        )  # Green color
                        if bbox_mesh is not None:
                            geometries.append(bbox_mesh)

            # Visualize all point clouds together
            o3d.visualization.draw_geometries(
                geometries,
                window_name="All Point Clouds",
                width=1024,
                height=768,
                point_show_normal=False,
                mesh_show_wireframe=False,
                mesh_show_back_face=False,
            )

            print(f"Visualized {len(geometries)-1} point clouds together")

        except Exception as e:
            print(f"Error visualizing all point clouds: {e}")
            import traceback

            traceback.print_exc()

    def _visualize_mask(self, rgb_image, mask, point, index):
        """Visualize segmentation mask"""
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Create overlay
        overlay = bgr_image.copy()
        mask_uint8 = (mask * 255).astype(np.uint8)
        overlay[mask_uint8 > 0] = (0, 255, 0)  # Green overlay
        vis = cv2.addWeighted(bgr_image, 0.7, overlay, 0.3, 0)

        # Draw clicked point
        cv2.circle(vis, point, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            vis,
            f"Object {index+1}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        window_name = f"Mask {index+1}"
        cv2.imshow(window_name, vis)
        cv2.waitKey(1000)  # Show for 1 second

    def _visualize_bbox(self, rgb_image, bbox, index):
        """Visualize oriented bounding box"""
        if bbox is None:
            return

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Get bbox corners in world coordinates
        corners = bbox.get_box_points()

        # Transform corners to camera coordinates
        corners_cam = (
            np.linalg.inv(self._cam2world[:3, :3])
            @ (corners - self._cam2world[:3, 3]).T
        ).T

        # Project to image coordinates
        fx = self._cam_intrinsic[0, 0]
        fy = self._cam_intrinsic[1, 1]
        cx = self._cam_intrinsic[0, 2]
        cy = self._cam_intrinsic[1, 2]

        # Project 3D points to 2D
        u = cx + fx * corners_cam[:, 0] / corners_cam[:, 2]
        v = cy + fy * corners_cam[:, 1] / corners_cam[:, 2]

        # Draw bbox edges
        for i in range(4):
            # Bottom face
            cv2.line(
                bgr_image,
                (int(u[i]), int(v[i])),
                (int(u[(i + 1) % 4]), int(v[(i + 1) % 4])),
                (255, 0, 0),
                2,
            )
            # Top face
            cv2.line(
                bgr_image,
                (int(u[i + 4]), int(v[i + 4])),
                (int(u[((i + 1) % 4) + 4]), int(v[((i + 1) % 4) + 4])),
                (255, 0, 0),
                2,
            )
            # Vertical edges
            cv2.line(
                bgr_image,
                (int(u[i]), int(v[i])),
                (int(u[i + 4]), int(v[i + 4])),
                (255, 0, 0),
                2,
            )

        cv2.putText(
            bgr_image,
            f"BBox {index+1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        window_name = f"Oriented BBox {index+1}"
        cv2.imshow(window_name, bgr_image)
        cv2.waitKey(1000)  # Show for 1 second

    def _visualize_all_masks(self, rgb_image, masks, clicked_points):
        """Visualize all segmentation masks in a single image"""
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Create a color palette for different objects
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

        # Create overlay with all masks
        overlay = bgr_image.copy()
        for i, (mask, point) in enumerate(zip(masks, clicked_points)):
            if mask is not None:
                # Get color for this object (cycle through colors)
                color = colors[i % len(colors)]

                # Ensure mask has the same shape as the image
                if mask.shape != (bgr_image.shape[0], bgr_image.shape[1]):
                    # Resize mask to match image dimensions
                    mask_resized = cv2.resize(
                        mask[-1].astype(np.float32),
                        (bgr_image.shape[1], bgr_image.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask = mask_resized > 0.5  # Threshold to get binary mask

                # Create colored mask overlay
                mask_uint8 = (mask * 255).astype(np.uint8)
                colored_mask = np.zeros_like(bgr_image)
                colored_mask[mask_uint8 > 0] = color

                # Blend with original image
                mask_alpha = 0.3
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, mask_alpha, 0)

                # Draw clicked point
                cv2.circle(
                    overlay, point, radius=5, color=(255, 255, 255), thickness=-1
                )
                cv2.putText(
                    overlay,
                    f"{i+1}",
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Add title
        cv2.putText(
            overlay,
            f"All Masks ({len(masks)} objects)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        window_name = "SAM2 All Masks"
        cv2.imshow(window_name, overlay)
        cv2.waitKey(1000)  # Show for 1 second

    def _visualize_all_bboxes(self, rgb_image, bboxes, clicked_points):
        """Visualize all bounding boxes in a single image"""
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        drawn = 0
        for i, bbox in enumerate(bboxes):
            if bbox is None:
                continue

            # OBB pose (local -> world)
            T_local_to_world = np.eye(4, dtype=float)
            T_local_to_world[:3, :3] = bbox.R
            T_local_to_world[:3, 3] = bbox.center

            # Local min/max from extent
            half = 0.5 * np.asarray(bbox.extent, dtype=float)
            bbox_min_max_local = np.vstack([-half, +half])  # (2,3)

            # World -> Cam (self._cam2world is cam->world)
            T_world_to_cam = np.linalg.inv(self._cam2world)
            T_local_to_cam = T_world_to_cam @ T_local_to_world

            # Draw onto bgr_image (in-place)
            draw_posed_3d_box(
                self._cam_intrinsic, bgr_image, T_local_to_cam, bbox_min_max_local
            )
            bgr_image = draw_xyz_axis(
                bgr_image, T_local_to_cam, scale=0.1, K=self._cam_intrinsic
            )
            drawn += 1

        cv2.putText(
            bgr_image,
            f"All Bounding Boxes ({drawn} objects)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("SAM2 All Bounding Boxes", bgr_image)
        cv2.waitKey(1000)  # non-blocking

    def _save_segmentation_results(self, index, mask, point_cloud, bbox):
        """Save segmentation results to files"""
        try:
            # Save mask
            mask_path = os.path.join(self._save_dir, "masks", f"mask_{index:03d}.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

            # Save point cloud
            if point_cloud is not None:
                pcd_path = os.path.join(
                    self._save_dir, "point_clouds", f"pcd_{index:03d}.ply"
                )
                o3d.io.write_point_cloud(pcd_path, point_cloud)

            # Save bounding box info
            if bbox is not None:
                bbox_path = os.path.join(
                    self._save_dir, "bboxes", f"bbox_{index:03d}.txt"
                )
                with open(bbox_path, "w") as f:
                    f.write(f"Center: {bbox.center}\n")
                    f.write(f"Extent: {bbox.extent}\n")
                    f.write(f"Rotation: {bbox.R}\n")

        except Exception as e:
            print(f"Error saving results: {e}")

    def _publish_results(self, rgbd_data):
        """Publish segmentation results"""
        try:
            # Publish bounding box data
            superellipsoids_data = []
            for i, bbox in enumerate(self._oriented_bboxes):
                if bbox is not None:
                    a1, a2, a3, e1, e2 = bb2superellipsoids(
                        bbox.extent[0], bbox.extent[1], bbox.extent[2], margin=1.1
                    )
                    # print(f"Original bbox extents (m): {bbox.extent[0]}, {bbox.extent[1]}, {bbox.extent[2]}")
                    # print(f"Superellipsoid parameters for object {i+1}: a1={a1}, a2={a2}, a3={a3}, e1={e1}, e2={e2}")
                    superellipsoid_vector = np.concatenate(
                        [
                            bbox.center.flatten(),  # Center coordinates
                            bbox.R.flatten(),  # Rotation matrix (flattened)
                            [a1, a2, a3, e1, e2],  # Superellipsoid parameters
                        ]
                    )

                    # print(f"Superellipsoid vector for object {i+1}: {superellipsoid_vector}")

                    superellipsoids_data.append(superellipsoid_vector)


            # add table        
            if self._publish_fixed_table:
                a1, a2, a3, e1, e2 = bb2superellipsoids(
                        2, 2, 0.06, margin=1.1
                    )
                superellipsoid_vector = np.concatenate(
                    [
                        [0, 0, -0.08],  # Center coordinates
                        np.eye(3).flatten(),  # Identity rotation matrix
                        [a1, a2, a3, e1, e2],  # Superellipsoid parameters
                    ]
                )
                superellipsoids_data.append(superellipsoid_vector)

            # print(f"Publishing {len(superellipsoids_data)} superellipsoids")

            if len(superellipsoids_data) > 0:
                superellipsoids_array = np.array(superellipsoids_data)
                out_superellipsoids = NamedVecListData(
                    num_vecs=len(superellipsoids_data),
                    vec_dim=superellipsoids_array.shape[1],
                    name="oriented_bboxes",
                )
                out_superellipsoids.set_data(
                    rgbd_data.get_time(),
                    [f"superellipsoid_{i}" for i in range(len(superellipsoids_data))],
                    superellipsoids_array,
                )
                # print(f"Publishing {len(superellipsoids_data)} superellipsoids")
                if self.percep_pub_que_dict[self._superellipsoid_pub_channel]:
                    self.percep_pub_que_dict[self._superellipsoid_pub_channel].put(
                        out_superellipsoids
                    )

        except Exception as e:
            print(f"Error publishing results: {e}")

    def _reset_segmentation(self):
        """Reset segmentation state"""
        self._clicked_points = []
        self._masks = []
        self._point_clouds = []
        self._oriented_bboxes = []
        self._segmentation_complete = False
        print("Segmentation reset")

    def _init_gui(self):
        """Initialize OpenCV window and mouse callback"""

        print("Starting GUI initialization...")
        self.clicked_point = None

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        cv2.resizeWindow(self._window_name, 640, 480)

    def _mouse_callback(self, event, x, y, flags, param=None):
        """Mouse callback for clicking on objects"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print(f"Clicked pixel: ({x}, {y})")
