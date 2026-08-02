import time

import pyrealsense2 as rs

import numpy as np
import cv2

from sensor.BaseSensor import BaseSensor
from data_type.basic_types.RGBDData import RGBDData


class RealSense(BaseSensor):
    """
    RealSense camera class.
    """

    def __init__(
        self,
        name: str,
        serial: str,
        *args,
        height: int = 480,
        width: int = 640,
        frequency: int = 30,
        visualize: bool = False,
        visual_preset: str | None = None,
        log_latency: bool = False,
        **kwargs
    ):
        super().__init__(name, *args, sensor_info_flag=True, **kwargs)
        self._serial = serial
        self._width = width
        self._height = height
        self._frequency = frequency
        self._visualize = visualize
        self._latest_vis = None  # latest composite frame for the main-thread pump
        # D400 depth-sensor visual preset, e.g. "High Density" / "High Accuracy"
        # (None => leave the camera default). Applied after the pipeline starts.
        self._visual_preset = visual_preset
        self._window_name = f"RealSense [{name}] (press 'q' to close window)"
        # Print the SENSOR-SIDE build latency (frame-arrival -> queued for publish:
        # align + cvtColor + RGBDData pack). Subtract this from the capture->receive
        # delay reported by scripts/tools/lcm_image_delay.py to split sensor-side
        # processing from serialize+transport.
        self._log_latency = log_latency
        self._proc_ms = []  # rolling per-frame build times (ms)

        # rs related
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(serial)
        self._config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, frequency
        )
        self._config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, frequency
        )
        align_to = rs.stream.color
        self._rs_align = rs.align(align_to)

    def _run(self):
        """
        Run the RealSense camera.
        """
        # Start the pipeline
        profile = self._pipeline.start(self._config)
        self._apply_visual_preset(profile)

        # Wait for frames
        while not self._stop_event.is_set():
            frames = self._pipeline.wait_for_frames()
            # Frame-arrival time (wait_for_frames blocks until a frame is ready,
            # so this is when it landed, not idle time). Used only for the
            # sensor-side build-latency report.
            t_arrival = self.get_time()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if (not depth_frame) or (not color_frame):
                continue

            # Align the depth frame to color frame
            aligned_frames = self._rs_align.process(frames)
            # Get aligned frames
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # realsense rgb comes in 8bit unsign int
            # realsense depth comes in 16bit unsigned int
            data = RGBDData(
                height=self._height,
                width=self._width,
                rgb_channel_type=1,
                depth_channel_type=3,
                name=self.name,
            )
            data.set_data(
                t=self.get_time(),
                rgb_channel_type=1,
                rgb_image=rgb_image,
                depth_channel_type=3,
                depth_image=depth_image,
            )

            self.pub_data_queue.put(data)

            if self._log_latency:
                self._proc_ms.append((self.get_time() - t_arrival) * 1e3)
                if len(self._proc_ms) >= 30:
                    a = np.asarray(self._proc_ms)
                    print(
                        f"[RealSense {self.name}] sensor-side build "
                        f"mean {a.mean():.1f} ms  max {a.max():.1f} ms "
                        f"(align+cvtColor+pack, before publish)"
                    )
                    self._proc_ms.clear()

            if self._visualize:
                self._show_frames(color_image, depth_image)

    def _apply_visual_preset(self, profile, retries=5, retry_delay=0.3):
        """Set the depth sensor's visual preset (e.g. "High Density") by name.

        librealsense exposes presets as an enumerated option whose integer
        values differ across firmware/SDK versions, so we match by the option's
        human-readable value description rather than hard-coding the enum int.

        The preset is a hardware XU control; when two cameras start streaming
        concurrently their XU accesses race on the shared USB backend and the
        write can fail with "get_xu(...) Device or resource busy". That is
        transient, so retry a few times with a short backoff before giving up.
        """
        if not self._visual_preset:
            return
        want = self._visual_preset.lower().replace(" ", "").replace("_", "")
        for attempt in range(1, retries + 1):
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                if not depth_sensor.supports(rs.option.visual_preset):
                    print(f"[RealSense {self.name}] device does not support visual_preset")
                    return
                rng = depth_sensor.get_option_range(rs.option.visual_preset)
                for i in range(int(rng.min), int(rng.max) + 1):
                    desc = depth_sensor.get_option_value_description(
                        rs.option.visual_preset, i
                    )
                    if desc and desc.lower().replace(" ", "").replace("_", "") == want:
                        depth_sensor.set_option(rs.option.visual_preset, i)
                        print(f"[RealSense {self.name}] visual preset -> {desc}"
                              + (f" (attempt {attempt})" if attempt > 1 else ""))
                        return
                print(f"[RealSense {self.name}] unknown visual preset "
                      f"'{self._visual_preset}'; leaving camera default")
                return
            except Exception as e:
                if attempt < retries:
                    print(f"[RealSense {self.name}] set visual preset busy "
                          f"(attempt {attempt}/{retries}), retrying: {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"[RealSense {self.name}] failed to set visual preset "
                          f"'{self._visual_preset}' after {retries} attempts: {e}")

    def _show_frames(self, color_image, depth_image):
        """Buffer the latest color+depth composite for the MAIN thread to draw.

        OpenCV HighGUI is not thread-safe: calling imshow/waitKey from each
        sensor thread (one per camera) deadlocks the shared GUI backend, so with
        two cameras the windows hang. Instead each sensor thread only stores its
        latest frame here; the sensor MANAGER pumps imshow/waitKey for all
        cameras on the main thread (see BaseSensorManager._pump_visualization).
        """
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        # color_image is BGR (stream is bgr8), so it is imshow-ready as-is.
        combined = np.hstack((color_image, depth_vis))
        # Single-writer (this sensor thread) / single-reader (main thread); a
        # plain reference swap is atomic under the GIL, no lock needed.
        self._latest_vis = combined

    def get_latest_vis(self):
        """(window_name, image) for the main-thread display pump, or None."""
        img = getattr(self, "_latest_vis", None)
        if img is None:
            return None
        return self._window_name, img
