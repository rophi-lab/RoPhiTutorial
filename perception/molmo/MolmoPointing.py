import os
import time
import gc

import numpy as np
import cv2
import copy

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from data_type.basic_types.Point3DData import Point3DData
from data_type.basic_types.CameraInfoData import CameraInfoData
from perception.BasePerception import BasePerception
from utils.mujoco.camera import (
    get_camera_extrinsics,
    get_camera_intrinsics,
)
from utils.perception.camera import convert_pixel_to_world
from utils.lie.se3 import invSE3
from utils.molmo.pixel_extraction import extract_pixel_from_text
from utils.visualization.depth import normalize_depth_image
from utils.terminal.CommandLineListener import CommandLineListener


class MolmoPointing(BasePerception):
    def __init__(self, config):
        super().__init__(config)
        self.name = "MolmoPointing"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config["torch_dtype"] == "float32":
            self.torch_dtype = torch.float32
        elif config["torch_dtype"] == "float16":
            self.torch_dtype = torch.float16

        self._preload_model = config["preload_model"]
        self._model_loaded = False
        self._model = None
        if self._preload_model:
            self._model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map=config["device_map"],
            )
            self._model_loaded = True

        self._processor = AutoProcessor.from_pretrained(
            config["model_name"],
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=config["device_map"],
        )

        self._generation_config = GenerationConfig(
            max_new_tokens=200, stop_strings="<|endoftext|>"
        )

        self._cmd_listener = CommandLineListener()
        self._update_only_new_cmd = config["update_only_new_cmd"]
        self._cmd_obj_name = ""
        self._new_cmd = False
        self._cmd_listener.start()

        self._rgbd_channel = config["sub_manager"]["rgbd_channel"]
        self._point_pub_channel = config["pub_manager"]["point_channel"]

        self._visualization_for_debug = config["visualization_for_debug"]

        self._cam_intrinsic = np.eye(3)
        self._cam2world = np.eye(4)
        self._depth_factor = 1.0
        self._vis_point_pixels = None

    def initialize(self):
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
                    cam_param_initialized = True
                    print("Camera intrinsic and extrinsic parameters initialized.")

    def stop(self):
        self._cmd_listener.stop()
        super().stop()

    def _process(self):
        new_cmd = self._cmd_listener.get_next_input()
        if new_cmd is not None:
            self._cmd_obj_name = new_cmd
            self._new_cmd = True
            print("[Perception] New command: Point to " + self._cmd_obj_name)

        # check and extract data from queue
        if not self.extr_sub_que_dict[self._rgbd_channel].empty():
            # pop out the rgbd data for real time tracking
            while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

            rgb_image = rgbd_data.get_rgb_image()
            depth_image = rgbd_data.get_depth_image()

            if not self._update_only_new_cmd:
                # temp solution: pop all data in the queue
                # to make sure we only process the latest data
                while not self.extr_sub_que_dict[self._rgbd_channel].empty():
                    rgbd_data = self.extr_sub_que_dict[self._rgbd_channel].get()

                rgb_image = rgbd_data.get_rgb_image()
                depth_image = rgbd_data.get_depth_image()
            if (
                not self._update_only_new_cmd and self._cmd_obj_name != ""
            ) or self._new_cmd:
                self._new_cmd = False
                with torch.no_grad():
                    # print("before: ", torch.cuda.memory_allocated() / 1024**2, "MB")
                    # if the model is not preloaded, load it to the device
                    if not self._preload_model:
                        # load model to device
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self._processor.tokenizer.name_or_path,
                            trust_remote_code=True,
                            torch_dtype=self.torch_dtype,
                            device_map={"": 0},
                        )
                        self._model_loaded = True
                    # print("loaded: ", torch.cuda.memory_allocated() / 1024**2, "MB")
                    # Convert the image to a fo rmat suitable for the model
                    inputs = self._processor.process(
                        images=rgb_image,
                        text="Point to " + self._cmd_obj_name,
                        return_tensors="pt",
                    )

                    inputs["images"] = inputs["images"].to(self.torch_dtype)

                    inputs = {
                        k: v.to(self._model.device).unsqueeze(0)
                        for k, v in inputs.items()
                    }

                    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
                    output = self._model.generate_from_batch(
                        inputs,
                        self._generation_config,
                        tokenizer=self._processor.tokenizer,
                    )

                    # only get generated tokens; decode them to text
                    generated_tokens = output[0, inputs["input_ids"].size(1) :]
                    generated_text = self._processor.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                # get image size
                img_h, img_w, _ = np.shape(rgb_image)
                # print(f"Image size: {img_w} x {img_h}")
                # Extract the object name from the generated text.
                # molmo normalized the point to 0-100,
                # so we need to use w and h to convert it
                # back to pixel coordinates.
                # point_pixels is a list of tuples (x, y)
                point_pixels = extract_pixel_from_text(generated_text, img_h, img_w)
                self._vis_point_pixels = point_pixels
                if point_pixels is not None:
                    # Convert pixel coordinates to world coordinates
                    point_world = convert_pixel_to_world(
                        point_pixels[0],
                        depth_image,
                        self._cam_intrinsic,
                        self._cam2world,
                        depth_factor=self._depth_factor,
                        inverse_z_direction=False,
                    )
                    # point_world = []
                    # for pixel in point_pixels:
                    #     pixel = (int(float(pixel[0])), int(float(pixel[1])))
                    #     point_world.append(
                    #         convert_pixel_to_world(
                    #             pixel, depth_image, self._cam_intrinsic, self._cam2world
                    #         )
                    #     )
                    if point_world is None:
                        print("Invalid depth value at pixel location.")
                    else:
                        # Create Point3DData object and publish it
                        out_pt = Point3DData(num_points=1)
                        out_pt.set_time(rgbd_data.get_time())
                        out_pt.set_position(point_world.reshape(1, 3))
                        # print(point_world)
                        # Publish result (can be original + keypoints or just keypoints)
                        if self.percep_pub_que_dict[self._point_pub_channel]:
                            self.percep_pub_que_dict[self._point_pub_channel].put(
                                out_pt
                            )

                # remove model if we do not wish to preload it (to save gpu memory)
                if not self._preload_model:
                    # print(torch.cuda.memory_allocated() / 1024**2, "MB")
                    # print(torch.cuda.memory_reserved())
                    del self._model

                    gc.collect()
                    torch.cuda.empty_cache()
                    # print(torch.cuda.memory_allocated() / 1024**2, "MB")
                    # print(torch.cuda.memory_reserved())

            # if visualize debug
            if self._visualization_for_debug:
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                depth_display = normalize_depth_image(depth_image)
                bgr_image_copy = copy.deepcopy(bgr_image)
                if self._vis_point_pixels is not None:
                    # Draw the keypoints on the image
                    for pixel in self._vis_point_pixels:
                        pixel = (int(float(pixel[0])), int(float(pixel[1])))
                        cv2.circle(
                            bgr_image_copy,
                            pixel,
                            5,
                            (0, 255, 0),
                            -1,
                        )

                cv2.imshow(f"image ({self._rgbd_channel})", bgr_image)
                cv2.imshow(f"detection ({self._rgbd_channel})", bgr_image_copy)
                cv2.imshow(f"depth ({self._rgbd_channel})", depth_display)
                cv2.waitKey(1)
