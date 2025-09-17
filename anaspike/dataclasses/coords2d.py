from typing import Union

import numpy as np
from numpy.typing import NDArray, ArrayLike

from ..hdf5_mixin import HDF5Mixin



class Coords2D(HDF5Mixin):
    def __init__(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-dimensional arrays.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        self.__x = x
        self.__y = y

    @property
    def x(self) -> NDArray[np.float64]:
        return self.__x

    @property
    def y(self) -> NDArray[np.float64]:
        return self.__y

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, s: Union[int, slice, NDArray[np.int64]]) -> "Coords2D":
        if isinstance(s, int):
            return Coords2D(self.__x[s:s+1], self.__y[s:s+1])
        return Coords2D(self.__x[s], self.__y[s])

