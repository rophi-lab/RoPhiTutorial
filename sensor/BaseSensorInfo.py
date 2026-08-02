from abc import ABC, abstractmethod


class BaseSensorInfo(ABC):
    """
    Base class for sensor information.
    """

    def __init__(self, name: str, *args, fixed: bool = True, **kwargs):
        self.name = name
        self.fixed = fixed
        self.fk_data = None
        self.pub_data_queue = None

    @abstractmethod
    def update(self):
        """
        Update the sensor information.
        """
        raise NotImplementedError(
            f"Sensor {self.name} does not implement the update method."
        )

    @abstractmethod
    def update_extrinsic(self, fk_data):
        """
        This function should take in the latest FK data and update the extrinsic matrix.

        """
        raise NotImplementedError(
            f"Sensor {self.name} does not implement the update_extrinsic method."
        )

    def set_pub_data_queue(self, pub_data_queue):
        """
        Set the data queue for the sensor.
        """
        self.pub_data_queue = pub_data_queue
