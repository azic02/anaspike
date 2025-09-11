import numpy as np

from ...dataclasses.spike_train import SpikeTrainArray
from ...dataclasses.histogram import Histogram
from ...dataclasses.histogram import ContigBins
from ...dataclasses.interval import Interval
from ...functions.spike_derived_quantities import spike_counts


def construct_spike_time_histogram(spike_trains: SpikeTrainArray, t_bins: ContigBins) -> Histogram:
    per_neuron_histograms = (Histogram.construct_by_counting(t_bins, st) for st in spike_trains)
    all_neurons_histogram = Histogram(t_bins, np.sum([h.counts for h in per_neuron_histograms], axis=0))
    return all_neurons_histogram

def construct_interspike_interval_histogram(spike_trains: SpikeTrainArray, t_bins: ContigBins) -> Histogram:
    inter_spike_intervals = (np.diff(st) for st in spike_trains)
    per_neuron_histograms = (Histogram.construct_by_counting(t_bins, isi) for isi in inter_spike_intervals)
    all_neurons_histogram = Histogram(t_bins, np.sum([h.counts for h in per_neuron_histograms], axis=0))
    return all_neurons_histogram

def calculate_active_neuron_fraction(spike_trains: SpikeTrainArray, t_interval: Interval, thresh: int=1) -> float:
    counts = np.squeeze(spike_counts(spike_trains, t_interval))
    if counts.shape != (spike_trains.n_neurons, ):
        raise ValueError(f"Shape mismatch: counts.shape: {counts.shape}, (spike_trains.n_neurons, ): ({spike_trains.n_neurons}, )")
    active_mask = counts >= thresh
    n_active = np.sum(active_mask)
    n_total = spike_trains.n_neurons
    return float(n_active) / float(n_total)

