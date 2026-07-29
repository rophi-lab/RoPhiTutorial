import copy

import numpy as np

from data_type.BaseData import BaseData


class NamedVecListData(BaseData):
    """Named list of fixed-dim vectors (e.g. grasp / fingertip markers)."""

    def __init__(self, num_vecs: int = 0, vec_dim: int = 3, name: str = "named_vec_list"):
        super().__init__(name)
        self.num_vecs = int(num_vecs)
        self.vec_dim = int(vec_dim)
        self.name_list: list[str] = []
        self.vec_list = np.zeros((self.num_vecs, self.vec_dim), dtype=np.float64)

    def set_data(self, t: float, name_list: list, vec_list: np.ndarray) -> None:
        self.timestamp = float(t)
        self.name_list = copy.deepcopy(list(name_list))
        self.num_vecs = len(self.name_list)
        self.vec_list = np.asarray(vec_list, dtype=np.float64).reshape(
            self.num_vecs, -1
        )
        self.vec_dim = int(self.vec_list.shape[1]) if self.num_vecs else self.vec_dim

    def get_data(self):
        return (
            self.timestamp,
            copy.deepcopy(self.name_list),
            self.vec_list.copy(),
        )

    def get_vec_dim(self) -> int:
        return int(self.vec_dim)
