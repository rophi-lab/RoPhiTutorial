import numpy as np


class BaseShapePrimitives:
    def __init__(
        self, p: np.ndarray, R: np.ndarray, name: str = "base_shape_primitives"
    ):
        """
        Initialize the BaseShapePrimitives object.
        """
        self.name = name
        self.p = p
        self.R = R

        self.param_dim = None  # This should be set in subclasses

    def set_position(self, p: np.ndarray):
        self.p = p

    def set_orientation(self, R: np.ndarray):
        self.R = R

    def set_params(self):
        """
        Set the parameters of the shape primitive.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_position(self) -> np.ndarray:
        return self.p

    def get_orientation(self) -> np.ndarray:
        return self.R

    def get_dict_params(self) -> dict:
        """
        Get the parameters of the shape primitive.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_vectorized_params(self) -> np.ndarray:
        """
        Get the parameters of the shape primitive as a vector.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_approx_sign_distance(self, x: np.ndarray) -> np.ndarray:
        """
        Get the approximate signed distance from a point to the shape primitive.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_approx_nearest_surface_points(self, x: np.ndarray) -> np.ndarray:
        """
        Get the approximate nearest surface points on the shape primitive.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_surface_normal(self, surface_points: np.ndarray) -> np.ndarray:
        """
        Get the surface normal at a point on the shape primitive.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
