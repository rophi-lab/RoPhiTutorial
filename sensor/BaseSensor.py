import time
import threading

from abc import ABC, abstractmethod


class BaseSensor(ABC):
    """
    Base class for all sensors.
    """

    def __init__(self, name: str, *args, sensor_info_flag: bool = False, **kwargs):
        self.name = name

        self.sub_data_queue = None
        self.pub_data_queue = None

        self._stop_event = threading.Event()
        self._sensor_thread = threading.Thread(target=self._run, daemon=False)

        self.sensor_info_flag = sensor_info_flag

    def start(self):
        """
        Start the sensor.
        """
        # Create a thread to run the sensor
        self._sensor_thread.start()
        print(f"{self.name} started.")

    def stop(self):
        """
        Stop the sensor.
        """
        self._stop_event.set()
        self._sensor_thread.join()
        print(f"{self.name} closed.")

    def get_time(self):
        """
        Get the current time.
        """
        return time.time()

    def has_sensor_info(self):
        """
        Check if the sensor has sensor info.
        @return: True if the sensor has sensor info, False otherwise.
        """
        return self.sensor_info_flag

    def set_sub_data_queue(self, sub_data_queue):
        """
        Set the data queue for the sensor.
        @param[in] sub_data_queue: The data queue to set.
        """
        self.sub_data_queue = sub_data_queue

    def set_pub_data_queue(self, pub_data_queue):
        """
        Set the data queue for the sensor.
        @param[in] pub_data_queue: The data queue to set.
        """
        self.pub_data_queue = pub_data_queue

    @abstractmethod
    def _run(self):
        """
        Run the sensor.
        """
        raise NotImplementedError(
            f"Sensor {self.name} does not implement the _run method."
        )
