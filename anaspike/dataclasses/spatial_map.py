from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .coords2d import Coords2D



CoordsT = TypeVar("CoordsT", bound=Coords2D)
ElmnT = TypeVar("ElmnT", bound=np.generic)
class SpatialMap(Generic[CoordsT, ElmnT]):
    def __init__(self, coords: CoordsT, values: NDArray[ElmnT]):
        if values.ndim < 1:
            raise ValueError("values must at least be a one-dimensional array.")
        if len(coords) != values.shape[0]:
            raise ValueError("length of coords must match length of values.")
       
        self.__coords = coords
        self.__values = values

    @property
    def coords(self) -> CoordsT:
        return self.__coords

    @property
    def values(self) -> NDArray[ElmnT]:
        return self.__values

    @property
    def n(self) -> int:
        return self.values.shape[0]

    @property
    def shape(self):
        return self.values.shape[1:] if self.values.ndim > 1 else ()

    @property
    def ndim(self) -> int:
        return self.values.ndim - 1

