import numpy as np

from data_type.BaseData import BaseData


class Point3DData(BaseData):
    def __init__(self, num_points: int, name: str = "point_pos_data"):
        """
        Initialize the PointPosData object.
        """
        super().__init__(name)
        self.num_points = num_points
        self.position = np.zeros((num_points, 3))

    def set_data(self, timestamp: float, num_points: int, position: np.ndarray):
        """
        Set the data of the points.
        :param timestamp: The timestamp of the data.
        :param num_points: The number of points.
        :param position: A numpy array of shape (num_points, 3) representing the positions of the points.
        """
        self.timestamp = timestamp
        self.set_num_points(num_points)
        self.set_position(position)

    def set_num_points(self, num_points: int):
        """
        Set the number of points.
        :param num_points: The number of points.
        """
        if num_points <= 0:
            raise ValueError("Number of points must be positive.")
        self.num_points = num_points
        self.position = np.zeros((num_points, 3))
        # print("Number of points set to:", num_points, "and position array reset.")

    def set_position(self, position: np.ndarray):
        """
        Set the position of the points.
        :param position: A numpy array of shape (num_points, 3) representing the positions of the points.
        """
        if position.shape != (self.num_points, 3):
            raise ValueError(f"Position must be of shape ({self.num_points}, 3)")
        self.position = position.copy()

    def get_data(self):
        """
        Get the data of the points.
        :return: A dictionary containing the number of points and their positions.
        """
        return self.timestamp, self.num_points, self.position

    def get_num_points(self) -> int:
        """
        Get the number of points.
        :return: The number of points.
        """
        return self.num_points

    def get_position(self) -> np.ndarray:
        """
        Get the position of the points.
        :return: A numpy array of shape (num_points, 3) representing the positions of the points.
        """
        return self.position.copy()
