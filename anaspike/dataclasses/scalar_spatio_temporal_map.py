from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .spatio_temporal_map import SpatioTemporalMap
from .scalar_temporal_map import ScalarTemporalMap
from .scalar_temporal_map import correlation as temporal_map_correlation
from .cartesian_map_2d import CartesianMap2D
from .coords2d import Coords2D



CoordsT = TypeVar("CoordsT", bound=Coords2D)
class ScalarSpatioTemporalMap(SpatioTemporalMap[CoordsT, np.float64]):
    def __init__(self, coords: CoordsT, times: NDArray[np.float64], values: NDArray[np.float64]):
        super().__init__(coords, times, values)
        if self.ndim != 0:
            raise ValueError("ScalarSpatioTemporalMap only supports scalar values (i.e., `values` must have shape (n_coords, n_times)).")


def temporal_correlation(stm: ScalarSpatioTemporalMap[Coords2D],
                         ref: ScalarTemporalMap) -> CartesianMap2D[np.float64]:
    return CartesianMap2D(stm.coords,
                          np.array([temporal_map_correlation(ref, tm) for tm in stm.iter_coords_dim()]))


def temporal_correlation_matrix(stm: ScalarSpatioTemporalMap[Coords2D]) -> NDArray[np.float64]:
    return np.corrcoef(stm.values_coords_major, dtype=np.float64)

