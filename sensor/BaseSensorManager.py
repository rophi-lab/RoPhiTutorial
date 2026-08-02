from abc import ABC, abstractmethod

from sensor.BaseSensor import BaseSensor

from collections import defaultdict
from queue import Queue

import time
import threading


class BaseSensorManager(ABC):
    """
    Base class for sensor managers.
    """

    def __init__(self, config, *args, **kwargs):
        """
        Initialize the sensor manager.
        """
        self.num_sensors = config["num_sensors"]

        self.fk_sub_queue = Queue()

        self.sub_que_dict = defaultdict(Queue)

        # Each sensor should be a dictionary
        # with the key as the sensor name and
        # the value as the sensor object.
        # We use the sensor publish channel name in config
        # as the key.
        self._sensors = {}
        self._sensor_info = {}

        self._info_pub_dt = 1.0 / config["info_pub_freq"]
        self._info_last_pub_time = time.time()

        self._finished = False

    def start(self):
        """
        Start the sensor manager.
        Each sensor is supposed to run it's own thread.
        """
        # create different sensor module

        # run the sensor threads
        for sensor in self._sensors.values():
            sensor.start()

        print("Real sensor manager started.")

        try:
            while not self._finished:
                self._run_sensor_info()
                # cv2 HighGUI is single-threaded: draw ALL camera windows here on
                # the main thread (sensor threads only buffer their frames), so
                # multiple cameras don't deadlock the shared GUI backend.
                self._pump_visualization()
                time.sleep(self._info_pub_dt * 0.1)
        except KeyboardInterrupt:
            self._finished = True
            self.stop()
            print("Keyboard interrupt received. Stopping sensor manager.")
        finally:
            self._close_visualization()

    def _pump_visualization(self):
        """Main-thread cv2 display for every sensor that exposes get_latest_vis().
        A single waitKey pumps the shared GUI event loop for all windows."""
        drew = False
        for sensor in self._sensors.values():
            get_vis = getattr(sensor, "get_latest_vis", None)
            if get_vis is None:
                continue
            vis = get_vis()
            if vis is None:
                continue
            name, img = vis
            import cv2
            cv2.imshow(name, img)
            drew = True
        if drew:
            import cv2
            cv2.waitKey(1)

    def _close_visualization(self):
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    def stop(self):
        """
        Stop the sensor manager.
        """
        for sensor in self._sensors.values():
            sensor.stop()
        print("Real sensor manager closed.")

    def set_sub_que_dict(self, sub_que_dict: dict):
        """
        Set the subscriber queue dictionary.
        @param[in] sub_que_dict: Subscriber queue dictionary.
        """
        self.sub_que_dict = sub_que_dict
        for channel_name, sensor in self._sensors.items():
            if channel_name in sub_que_dict:
                sensor.set_sub_data_queue(sub_que_dict[channel_name])
            else:
                print(
                    f"Sensor {channel_name} not found in subscriber queue dictionary."
                )

    def set_pub_que_dict(self, pub_data_queue_dict):
        """
        Set the data queue for the sensors.
        @param[in] data_queue_dict: The data queue dictionary to set.
        """
        for channel_name, sensor in self._sensors.items():
            # print("channel_name", channel_name)
            # print(pub_data_queue_dict)
            if channel_name in pub_data_queue_dict:
                print("setting pub data queue for sensor: ", channel_name)
                sensor.set_pub_data_queue(pub_data_queue_dict[channel_name])
                if sensor.has_sensor_info():
                    self._sensor_info[channel_name].set_pub_data_queue(
                        pub_data_queue_dict[channel_name + "_info"]
                    )
            else:
                print(
                    f"Sensor {channel_name} not found in publish data queue dictionary."
                )

    def set_fk_sub_queue(self, fk_sub_queue):
        """
        Set the FK subscription queue.
        @param[in] fk_sub_queue: The FK subscription queue to set.
        """
        self.fk_sub_queue = fk_sub_queue

    @abstractmethod
    def initialize_sensors_and_sensor_info(self):
        """
        This function should declare new sensor modules and sensor info
        and put them into self._sensors and self._sensor_info respectively.
        """
        raise NotImplementedError(
            "Sensor manager does not implement the _initialize_sensor_info method."
        )

    def _run_sensor_info(self):
        """
        Run the sensor info thread.
        """

        # subscribe to FK
        if not self.fk_sub_queue.empty():
            fk_data = self.fk_sub_queue.get()

            # publish FK data
            for _, sensor_info in self._sensor_info.items():
                sensor_info.update_extrinsic(fk_data)

        if time.time() - self._info_last_pub_time > self._info_pub_dt:
            # publish sensor info
            for _, sensor_info in self._sensor_info.items():
                sensor_info.update()

            # update last publish time
            self._info_last_pub_time = time.time()
