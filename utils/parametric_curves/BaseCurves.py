import numpy as np


class BaseCurves:
    def __init__(
        self,
        curve_type: str = "vmp",
        terminal_time: float = 5,
        space_type: str = "vector_space",
        dim: int = 6,
        *args,
        **kwargs,
    ):
        """
        Base class for parametric curves.
        :param curve_type: Type of the curve (e.g., "vmp", "cubic", etc.)
        :param terminal_time: The time at which the curve ends.
        :param space_type: Type of space (e.g., "vector_space", "matrix_space", etc.)
        :param dim: Dimension of the space.
        """
        self._curve_type = curve_type
        self._terminal_time = terminal_time
        self._space_type = space_type
        self._dim = dim
        self._parameters = {}

    def __str__(self):
        return f"Curve Type: {self._curve_type}, Terminal Time: {self._terminal_time}"

    def __call__(self, t: np.ndarray):
        """
        Returns the position of the curve at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of position values
        """
        return self._position(t)

    def _position(self, t):
        """
        Returns the position of the curve at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of position values
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def velocity(self, t):
        """
        Returns the velocity of the curve at time t.
        @param t: (n, 1) array of time values
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def acceleration(self, t):
        """
        Returns the acceleration of the curve at time t.
        @param t: (n, 1) array of time values
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def jerk(self, t):
        """
        Returns the jerk of the curve at time t.
        @param t: (n, 1) array of time values
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_trajectory(self, num_points=100):
        t = np.linspace(0, self._terminal_time, num_points).reshape(num_points, 1)
        return self(t)

    def get_velocity_trajectory(self, num_points=100):
        t = np.linspace(0, self._terminal_time, num_points).reshape(num_points, 1)
        return self.velocity(t)

    def get_acceleration_trajectory(self, num_points=100):
        t = np.linspace(0, self._terminal_time, num_points).reshape(num_points, 1)
        return self.acceleration(t)

    def get_jerk_trajectory(self, num_points=100):
        t = np.linspace(0, self._terminal_time, num_points).reshape(num_points, 1)
        return self.jerk(t)

    def get_parameters(self):
        """
        Returns the parameters of the curve.
        :return: Dictionary of parameters
        """
        return self._parameters

    def get_curve_type(self):
        """
        Returns the type of the curve.
        :return: Curve type
        """
        return self._curve_type

    def get_terminal_time(self):
        """
        Returns the terminal time of the curve.
        :return: Terminal time
        """
        return self._terminal_time

    def get_space_type(self):
        """
        Returns the type of space.
        :return: Space type
        """
        return self._space_type

    def get_dim(self):
        """
        Returns the dimension of the space.
        :return: Dimension
        """
        return self._dim
