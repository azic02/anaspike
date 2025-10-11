from typing import Generic, TypeVar, Iterator

import numpy as np
from numpy.typing import NDArray

from .coords2d import Coords2D
from .spatial_map import SpatialMap
from .temporal_map import TemporalMap
from .interval2d import Interval2D



CoordsT = TypeVar("CoordsT", bound=Coords2D)
ValT = TypeVar("ValT", bound=np.generic)
class SpatioTemporalMap(Generic[CoordsT, ValT]):
    def __init__(self, coords: CoordsT, times: NDArray[np.float64], values: NDArray[ValT]):
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional array.")
        if values.ndim < 2:
            raise ValueError("values must at least be a two-dimensional array (first dimension maps to coords, second dimension maps to times).")
        if len(coords) != values.shape[0]:
            raise ValueError("length of coords must match the first dimension of values.")
        if len(times) != values.shape[1]:
            raise ValueError("length of times must match the second dimension of values.")
       
        self.__coords = coords
        self.__times = times
        self.__values = values

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__values.shape[2:] if self.__values.ndim > 2 else ()

    @property
    def times(self) -> NDArray[np.float64]:
        return self.__times

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def coords(self) -> CoordsT:
        return self.__coords

    @property
    def n_values(self) -> int:
        return len(self.coords)

    @property
    def values_coords_major(self) -> NDArray[ValT]:
        return self.__values

    @property
    def values_time_major(self) -> NDArray[ValT]:
        return np.swapaxes(self.__values, 0, 1)

    def iter_coords_dim(self) -> Iterator[TemporalMap[ValT]]:
        for elmn in self.__values:
            yield TemporalMap(self.__times, elmn)

    def iter_time_dim(self) -> Iterator[SpatialMap[CoordsT, ValT]]:
        for t_idx in range(self.n_times):
            yield SpatialMap(self.__coords, self.__values[:, t_idx])

