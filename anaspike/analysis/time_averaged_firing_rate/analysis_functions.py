import numpy as np

from .time_averaged_firing_rate import TimeAveragedFiringRate
from ...dataclasses.histogram import Histogram
from ...dataclasses.histogram import ContigBins
from ...dataclasses.contig_bins_2d import ContigBins2D
from ...dataclasses.contig_bins_2d import calculate_bin_means as calculate_bin_means_2d
from ...dataclasses.grid import RectilinearGrid2D
from ...dataclasses.field import GridField2D



def mean(fr: TimeAveragedFiringRate) -> float:
    return np.mean(fr, dtype=np.float64)

def std(fr: TimeAveragedFiringRate) -> float:
    return np.std(fr, dtype=np.float64)

def construct_histogram(fr: TimeAveragedFiringRate, bins: ContigBins) -> Histogram:
    return Histogram.construct_by_counting(bins, fr.elements)

def bin_spatially(fr: TimeAveragedFiringRate, bins: ContigBins2D) -> GridField2D[RectilinearGrid2D, np.float64]:
    return calculate_bin_means_2d(bins, fr)

