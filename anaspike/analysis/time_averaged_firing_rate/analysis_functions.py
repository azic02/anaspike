import numpy as np

from .time_averaged_firing_rate import TimeAveragedFiringRate
from ...dataclasses.histogram import Histogram
from ...dataclasses.histogram import ContigBins



def mean(fr: TimeAveragedFiringRate) -> float:
    return np.mean(fr.as_nparray, dtype=np.float64)

def std(fr: TimeAveragedFiringRate) -> float:
    return np.std(fr.as_nparray, dtype=np.float64)

def construct_histogram(fr: TimeAveragedFiringRate, bins: ContigBins) -> Histogram:
    return Histogram.construct_by_counting(bins, fr.as_nparray)

