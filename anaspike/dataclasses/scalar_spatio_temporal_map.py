from typing import TypeVar, Iterator, cast

import numpy as np
from numpy.typing import NDArray

from .spatio_temporal_map import SpatioTemporalMap
from .scalar_temporal_map import ScalarTemporalMap
from .scalar_temporal_map import correlation as temporal_map_correlation
from .scalar_spatial_map import ScalarSpatialMap
from .scalar_spatial_map import morans_i as morans_i_sm
from .coords2d import Coords2D



CoordsT = TypeVar("CoordsT", bound=Coords2D)
class ScalarSpatioTemporalMap(SpatioTemporalMap[CoordsT, np.float64]):
    def __init__(self, coords: CoordsT, times: NDArray[np.float64], values: NDArray[np.float64]):
        super().__init__(coords, times, values)
        if self.ndim != 0:
            raise ValueError("ScalarSpatioTemporalMap only supports scalar values (i.e., `values` must have shape (n_coords, n_times)).")

    def iter_time_dim(self) -> Iterator[ScalarSpatialMap[CoordsT]]:
        yield from cast(Iterator[ScalarSpatialMap[CoordsT]], super().iter_time_dim())

    def iter_coords_dim(self) -> Iterator[ScalarTemporalMap]:
        yield from cast(Iterator[ScalarTemporalMap], super().iter_coords_dim())


def temporal_correlation(stm: ScalarSpatioTemporalMap[CoordsT],
                         ref: ScalarTemporalMap) -> ScalarSpatialMap[CoordsT]:
    return ScalarSpatialMap(stm.coords, np.array([temporal_map_correlation(ref, tm) for tm in stm.iter_coords_dim()]))


def temporal_correlation_matrix(stm: ScalarSpatioTemporalMap[Coords2D]) -> NDArray[np.float64]:
    return np.corrcoef(stm.values_coords_major, dtype=np.float64)


def morans_i(stm: ScalarSpatioTemporalMap[CoordsT]) -> ScalarTemporalMap:
    return ScalarTemporalMap(stm.times, np.array([morans_i_sm(sm) for sm in stm.iter_time_dim()]))

