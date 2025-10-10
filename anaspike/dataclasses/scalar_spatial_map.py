from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .spatial_map import SpatialMap
from .coords2d import Coords2D


CoordsT = TypeVar("CoordsT", bound=Coords2D)
class ScalarSpatialMap(SpatialMap[CoordsT, np.float64]):
    def __init__(self, coords: CoordsT, values: NDArray[np.float64]):
        if values.ndim != 1:
            raise ValueError("values must be a 1D array")
        super().__init__(coords, values)

