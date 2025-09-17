from typing import TypeVar

import numpy as np

from .time_averaged_firing_rate import TimeAveragedFiringRate
from ...dataclasses.histogram import Histogram
from ...dataclasses.bins import ContigBins1D
from ...dataclasses.bins import ContigBins2D
from ...dataclasses.bins import calculate_bin_means_2d
from ...dataclasses.grid import Grid1D, RectilinearGrid2D
from ...dataclasses.field import GridField2D



def mean(fr: TimeAveragedFiringRate) -> float:
    return np.mean(fr, dtype=np.float64)

def std(fr: TimeAveragedFiringRate) -> float:
    return np.std(fr, dtype=np.float64)

Grid1dT = TypeVar("Grid1dT", bound=Grid1D)
def construct_histogram(fr: TimeAveragedFiringRate, bins: ContigBins1D[Grid1dT]) -> Histogram:
    return Histogram.construct_by_counting(bins, fr.elements)

Grid2dT = TypeVar("Grid2dT", bound=RectilinearGrid2D)
def bin_spatially(fr: TimeAveragedFiringRate, bins: ContigBins2D[Grid2dT]) -> GridField2D[Grid2dT, np.float64]:
    return calculate_bin_means_2d(bins, fr)

