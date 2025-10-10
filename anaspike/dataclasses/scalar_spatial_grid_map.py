from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .spatial_grid_map import SpatialGridMap
from .grid import RectilinearGrid2D


GridT = TypeVar("GridT", bound=RectilinearGrid2D)
class ScalarSpatialGridMap(SpatialGridMap[GridT, np.float64]):
    def __init__(self, grid: GridT, values: NDArray[np.float64]):
        if values.ndim != 1:
            raise ValueError("values must be a 1D array")
        super().__init__(grid, values)

