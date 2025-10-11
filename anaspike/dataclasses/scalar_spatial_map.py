from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .spatial_map import SpatialMap
from .coords2d import Coords2D
from ..functions.statistical_quantities import morans_i as morans_i_untyped
from ..functions._helpers import calculate_pairwise_2d_euclidean_distances


CoordsT = TypeVar("CoordsT", bound=Coords2D)
class ScalarSpatialMap(SpatialMap[CoordsT, np.float64]):
    def __init__(self, coords: CoordsT, values: NDArray[np.float64]):
        if values.ndim != 1:
            raise ValueError("values must be a 1D array")
        super().__init__(coords, values)


def morans_i(sm: ScalarSpatialMap[CoordsT]) -> float:
    distances = calculate_pairwise_2d_euclidean_distances(sm.coords.x, sm.coords.y)
    np.fill_diagonal(distances, 1.)
    weights = np.power(distances, -1)
    return morans_i_untyped(sm.values, weights)

