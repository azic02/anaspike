from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .coords2d import Coords2D



ElmnT = TypeVar("ElmnT", bound=np.generic)
class CartesianMap2D(Generic[ElmnT]):
    def __init__(self, coords: Coords2D, elements: NDArray[ElmnT]):
        if elements.ndim < 1:
            raise ValueError("elements must at least be a one-dimensional array.")
        if len(coords) != elements.shape[0]:
            raise ValueError("length of coords must match length of elements.")
       
        self.__coords = coords
        self.__elements = elements

    @property
    def coords(self) -> Coords2D:
        return self.__coords

    @property
    def elements(self) -> NDArray[ElmnT]:
        return self.__elements

    @property
    def n(self) -> int:
        return self.elements.shape[0]

    @property
    def element_shape(self):
        return self.elements.shape[1:] if self.elements.ndim > 1 else ()

    @property
    def element_ndim(self) -> int:
        return self.elements.ndim - 1

