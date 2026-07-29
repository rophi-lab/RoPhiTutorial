import numpy as np

from data_type.BaseData import BaseData


class ColInfoData(BaseData):
    """
    This class is used to store the collision information.
    """

    def __init__(
        self,
        name: str = "static_col_info_data",
        dict_col_geoms: dict = None,
    ):
        """
        Initialize the ColInfoData object.
        """
        super().__init__(name)
        self.dict_col_geoms = dict_col_geoms if dict_col_geoms is not None else {}

    def reset_data(self):
        """
        Reset the collision information.
        """
        self.dict_col_geoms = {}
        self.timestamp = 0.0

    def add_col_geom(
        self, name: str, geom_type: str, size: np.ndarray, offset: np.ndarray
    ):
        """
        Add a collision geometry to the dictionary.
        @param[in] name: The name of the geometry.
        @param[in] geom_type: The type of the geometry.
        @param[in] size: The size of the geometry.
        @param[in] offset: The offset SE3 pose of the geometry.
        """
        self.dict_col_geoms[name] = {
            "type": geom_type,
            "size": size,
            "offset": offset,
            "size_dim": len(size),
        }

    def set_data(
        self,
        t: float,
        dict_col_geoms: dict,
    ):
        """
        Set the collision information.
        @param[in] t: The timestamp of the data.
        @param[in] dict_col_geoms: The dictionary of collision geometries.
        """
        self.dict_col_geoms = dict_col_geoms
        self.timestamp = t

    def get_data(self):
        """
        Get the collision information.
        @return: The dictionary of collision geometries.
        """
        return self.timestamp, self.dict_col_geoms
