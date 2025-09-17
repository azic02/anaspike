from typing import TypeVar

import numpy as np

from .spike_trains import SpikeTrains
from ...dataclasses.histogram import Histogram
from ...dataclasses.bins import ContigBins1D
from ...dataclasses.grid import Grid1D



Grid1dT = TypeVar("Grid1dT", bound=Grid1D)
def construct_spike_time_histogram(spike_trains: SpikeTrains,
                                   t_bins: ContigBins1D[Grid1dT]) -> Histogram:
    per_neuron_histograms = (Histogram.construct_by_counting(t_bins, st) for st in spike_trains)
    all_neurons_histogram = Histogram(t_bins, np.sum([h.counts for h in per_neuron_histograms], axis=0))
    return all_neurons_histogram

def construct_interspike_interval_histogram(spike_trains: SpikeTrains,
                                            t_bins: ContigBins1D[Grid1dT]) -> Histogram:
    inter_spike_intervals = (np.diff(st) for st in spike_trains)
    per_neuron_histograms = (Histogram.construct_by_counting(t_bins, isi) for isi in inter_spike_intervals)
    all_neurons_histogram = Histogram(t_bins, np.sum([h.counts for h in per_neuron_histograms], axis=0))
    return all_neurons_histogram

