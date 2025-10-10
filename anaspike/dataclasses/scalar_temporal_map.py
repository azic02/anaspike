import numpy as np
from numpy.typing import NDArray

from .temporal_map import TemporalMap



class ScalarTemporalMap(TemporalMap[np.float64]):
    def __init__(self, times: NDArray[np.float64], values: NDArray[np.float64]):
        super().__init__(times, values)
        if self.ndim != 0:
            raise ValueError("Only supports scalar values (i.e. `values` must be a one-dimensional array).")


def correlation(tm1: ScalarTemporalMap, tm2: TemporalMap[np.float64]) -> float:
    if tm1.n_times != tm2.n_times:
        raise ValueError("Both TemporalMaps must have the same number of time points.")
    return np.corrcoef(tm1.values, tm2.values)[0, 1]

