from typing import TypeVar

import numpy as np

from .firing_rates import FiringRates, BinnedFiringRates
from ..dataclasses.histogram import Histogram
from ..dataclasses.bins import ContigBins1D
from ..dataclasses.bins import ContigBins2D
from ..dataclasses.bins import calculate_bin_means_2d
from ..dataclasses.grid import Grid1D, RectilinearGrid2D, RegularGrid2D
from ..dataclasses.grid_map_2d import calculate_psd_2d, calculate_autocorrelation_2d_wiener_khinchin



def mean(fr: FiringRates) -> float:
    return np.mean(fr, dtype=np.float64)

def std(fr: FiringRates) -> float:
    return np.std(fr, dtype=np.float64)

Grid1dT = TypeVar("Grid1dT", bound=Grid1D)
def construct_histogram(fr: FiringRates, bins: ContigBins1D[Grid1dT]) -> Histogram:
    return Histogram.construct_by_counting(bins, fr.values)

Grid2dT = TypeVar("Grid2dT", bound=RectilinearGrid2D)
def bin_spatially(fr: FiringRates, bins: ContigBins2D[Grid2dT]) -> BinnedFiringRates[Grid2dT]:
    return calculate_bin_means_2d(bins, fr)

def calculate_spatial_psd(fr: BinnedFiringRates[RegularGrid2D]):
    return calculate_psd_2d(fr)

def calculate_spatial_autocorrelation_wiener_khinchin(fr: BinnedFiringRates[RegularGrid2D]):
    return calculate_autocorrelation_2d_wiener_khinchin(fr)

