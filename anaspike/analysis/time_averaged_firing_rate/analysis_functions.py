import numpy as np

from .time_averaged_firing_rate import TimeAveragedFiringRate
from ...dataclasses.histogram import Histogram
from ...dataclasses.histogram import ContigBins
from ...dataclasses.contig_bins_2d import ContigBins2D
from ...dataclasses.contig_bins_2d import calculate_bin_means as calculate_bin_means_2d
from ...dataclasses.coords2d import Coords2D



def mean(fr: TimeAveragedFiringRate) -> float:
    return np.mean(fr.as_nparray, dtype=np.float64)

def std(fr: TimeAveragedFiringRate) -> float:
    return np.std(fr.as_nparray, dtype=np.float64)

def construct_histogram(fr: TimeAveragedFiringRate, bins: ContigBins) -> Histogram:
    return Histogram.construct_by_counting(bins, fr.as_nparray)

def bin_spatially(fr: TimeAveragedFiringRate, coords: Coords2D, bins: ContigBins2D) -> TimeAveragedFiringRate:
    if len(fr.as_nparray) != len(coords):
        raise ValueError("Length of firing rates and coordinates must be the same.")
    return TimeAveragedFiringRate(np.reshape(np.transpose(calculate_bin_means_2d(bins, coords, fr.as_nparray)), -1))

