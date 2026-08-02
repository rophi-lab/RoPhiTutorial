import numpy as np
import cv2

from data_type.basic_types.NamedVecListData import NamedVecListData
from data_type.basic_types.Point3DData import Point3DData
from data_type.basic_types.CameraInfoData import CameraInfoData
from perception.BasePerception import BasePerception
from utils.perception.camera import convert_pixel_to_world
from utils.lie.se3 import invSE3


class ClickTwoAndPoints(BasePerception):
    def __init__(self, config):
        super().__init__(config)
        self.name = "ClickTwoAndPoints"
        # self._cmd_listener = None
        # self._update_only_new_cmd = config["update_only_new_cmd"]
        # self._cmd_obj_name = ""
        # self._new_cmd = False

        self._rgbd_channel = config["sub_manager"]["rgbd_channel"]
        # Use a new channel for two points output
        self._two_points_pub_channel = config["pub_manager"]["named_vec_list_channel"]

        # self._visualization_for_debug = config["visualization_for_debug"]

        self._cam_intrinsic = np.eye(3)
        self._cam2world = np.eye(4)
        self._depth_factor = 1.0
        self._vis_point_pixels = None

        self._clicked_pts = []  # Store two clicked points
        self._last_clicked_pts = []  # For visualization

        self._last_points_world_dir = []  # For publishing
        self._got_first_points = False

    def initialize(self):
        # initialize gui
        self._init_gui()

        # wait for intrinsic and extrinsic camera parameters
        cam_param_initialized = False
        while not cam_param_initialized:
            if not self.extr_sub_que_dict[self._rgbd_channel + "_info"].empty():
                cam_info = self.extr_sub_que_dict[self._rgbd_channel + "_info"].get()
                if isinstance(cam_info, CameraInfoData):
                    self._cam_intrinsic = cam_info.get_intrinsic()
                    self._cam2world = invSE3(cam_info.get_extrinsic())
                    self._depth_factor = cam_info.get_depth_factor()
                    # print(self._cam2world)
                    print(self._depth_factor)
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    def _process(self):
        """
        Display the frame and return clicked pixel coordinate if any.
        This is designed to be called in a while loop.
        """
        if not self.extr_sub_que_dict[self._rgbd_channel].empty():
            
            # pop out the rgbd data for real time tracking
            while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

            rgb_image = rgbd_data.get_rgb_image()
            depth_image = rgbd_data.get_depth_image()

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # Draw last clicked points for feedback
            for pt in self._last_clicked_pts:
                cv2.circle(
                    bgr_image,
                    pt,
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )
            # Draw current clicked points (not yet published)
            for pt in self._clicked_pts:
                cv2.circle(
                    bgr_image,
                    pt,
                    radius=5,
                    color=(0, 255, 0),
                    thickness=-1,
                )
            cv2.imshow(self._window_name, bgr_image)
            # depth_display = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            # )

            key = cv2.waitKey(1)  # Add key check if needed (e.g. for exit)

            if len(self._clicked_pts) == 2:
                pts = self._clicked_pts
                points_world = []
                valid = False
                if len(pts) == 2:
                    valid = True
                    for pt in pts:
                        point_world = convert_pixel_to_world(
                            pt,
                            depth_image,
                            self._cam_intrinsic,
                            self._cam2world,
                            depth_factor=self._depth_factor,
                            inverse_z_direction=False,
                        )
                        print(point_world)
                        if point_world is None:
                            print(f"Invalid depth value at pixel location {pt}.")
                            valid = False
                            self._clicked_pts = []  # Reset for next pair
                            break
                        points_world.append(point_world)
                if valid:

                    points_world_np = np.array(points_world)
                    # convert points world to a 9 by 1 vector by appending the direction vector
                    self._last_points_world_dir = np.vstack(
                        [
                            points_world_np,
                            np.zeros(3),
                            points_world_np[1, :],
                            points_world_np[0, :],
                            np.zeros(3),
                        ]
                    ).reshape(2, -1)

                    self._last_clicked_pts = pts.copy()
                    self._clicked_pts = []  # Reset for next pair
                    self._got_first_points = True
                    print("New 3D points: ", self._last_points_world_dir)
            if self._got_first_points:
                # Create NamedVecListData object and publish it
                out_data = NamedVecListData(
                    num_vecs=2, vec_dim=9, name="clicked_two_points"
                )
                name_list = ["point1", "point2"]
                vec_list = self._last_points_world_dir
                out_data.set_data(rgbd_data.get_time(), name_list, vec_list)
                if self.percep_pub_que_dict[self._two_points_pub_channel]:
                    self.percep_pub_que_dict[self._two_points_pub_channel].put(out_data)
                    # print(f"Published 3D points: {vec_list}")

    def _init_gui(self):
        """
        Initializes the OpenCV window and mouse callback once.
        Should be called once in __init__ or before _process.
        """
        self._clicked_pts = []
        self._last_clicked_pts = []
        self._window_name = "Click on Image"
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self._clicked_pts) < 2:
                self._clicked_pts.append((x, y))
                print(f"Clicked pixel: ({x}, {y})")
