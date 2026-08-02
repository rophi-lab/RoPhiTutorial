import numpy as np
import cv2

from data_type.basic_types.Point3DData import Point3DData
from data_type.basic_types.CameraInfoData import CameraInfoData
from data_type.basic_types.NamedVecListData import NamedVecListData
from perception.BasePerception import BasePerception
from utils.perception.camera import convert_pixel_to_world
from utils.lie.se3 import invSE3


class ClickAndPoint(BasePerception):
    def __init__(self, config):
        super().__init__(config)
        self.name = "ClickAndPoint"
        # self._cmd_listener = None
        # self._update_only_new_cmd = config["update_only_new_cmd"]
        # self._cmd_obj_name = ""
        # self._new_cmd = False

        self._rgbd_channel = config["sub_manager"]["rgbd_channel"]
        self._publisher_name = config["pub_manager"]["name"]
        if self._publisher_name == "point3d":
            self._point_pub_channel = config["pub_manager"]["point_channel"]
        elif self._publisher_name == "named_vec":
            self._point_pub_channel = config["pub_manager"]["named_vec_list_channel"]
        else:
            raise ValueError(f"Unknown publisher name: {self._publisher_name}")

        self._continous_mode = config["continous_mode"]

        # self._visualization_for_debug = config["visualization_for_debug"]

        self._cam_intrinsic = np.eye(3)
        self._cam2world = np.eye(4)
        self._depth_factor = 1.0
        self._vis_point_pixels = None

        self._last_clicked_pt = None
        self._got_first_point = False
        self._last_world_point = None

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

            if self._last_clicked_pt is not None:
                cv2.circle(
                    bgr_image,
                    self._last_clicked_pt,
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )
            cv2.imshow(self._window_name, bgr_image)
            # cv2.imshow("Depth Image", depth_image)

            key = cv2.waitKey(1)  # Add key check if needed (e.g. for exit)

            if self.clicked_point is not None:

                pt = self.clicked_point
                self.clicked_point = None  # Reset after read

                # Convert pixel coordinates to world coordinates
                point_world = convert_pixel_to_world(
                    pt,
                    depth_image,
                    self._cam_intrinsic,
                    self._cam2world,
                    depth_factor=self._depth_factor,
                    inverse_z_direction=False,
                )
                self._last_clicked_pt = pt

                if point_world is None:
                    print("Invalid depth value at pixel location.")
                else:
                    self._got_first_point = True
                    self._last_world_point = point_world
                    print(f"Updated 3D point: {point_world}")

            if self._got_first_point:
                if self._publisher_name == "point3d":
                    # Create Point3DData object and publish it
                    out_pt = Point3DData(num_points=1)
                    out_pt.set_time(rgbd_data.get_time())
                    out_pt.set_position(self._last_world_point.reshape(1, 3))

                    # Publish result (can be original + keypoints or just keypoints)
                    if self.percep_pub_que_dict[self._point_pub_channel]:
                        self.percep_pub_que_dict[self._point_pub_channel].put(out_pt)
                elif self._publisher_name == "named_vec":
                    # Create NamedVecListData object and publish it
                    out_data = NamedVecListData(
                        num_vecs=1, vec_dim=3, name="clicked_point"
                    )
                    out_data.set_data(
                        rgbd_data.get_time(),
                        ["clicked_point"],
                        self._last_world_point.reshape(1, 3),
                    )
                    if self.percep_pub_que_dict[self._point_pub_channel]:
                        self.percep_pub_que_dict[self._point_pub_channel].put(out_data)

            if not self._continous_mode:
                self._got_first_point = False
                self._last_world_point = None

    def _init_gui(self):
        """
        Initializes the OpenCV window and mouse callback once.
        Should be called once in __init__ or before _process.
        """
        self.clicked_point = None
        self._window_name = "Click on Image"
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print(f"Clicked pixel: ({x}, {y})")
