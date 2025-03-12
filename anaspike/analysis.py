from typing import Iterable, Sequence, Dict, Tuple, Literal
from itertools import product
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr, variation

from anaspike.dataclasses.nest_devices import PopulationData, NeuronData, SpikeRecorderData
from anaspike.dataclasses.interval import Interval, Bin
from anaspike.functions import (firing_rates,
                                spike_counts_in_spacetime_region,
                                pearson_correlation_offset_data)
from anaspike.hayleigh import get_autocorr, average_over_interp2d
from anaspike.sigrid import cross_correlate_masked, get_radial_acorr



def firing_rates_evolution(neurons: PopulationData, spike_recorder: SpikeRecorderData, t_bins: Iterable[Bin]) -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(neurons.ids)
    return np.array([firing_rates(spike_trains, t_bin) for t_bin in t_bins])


def time_avg_firing_rates_histogram(population: PopulationData, spike_recorder: SpikeRecorderData, t_interval: Interval, nbins: int) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    spike_trains = spike_recorder.get_spike_trains(population.ids)
    time_avg_firing_rates = firing_rates(spike_trains, t_interval)

    firing_rate_bins = Interval(min(time_avg_firing_rates), max(time_avg_firing_rates)).bin(nbins)
    bin_edges = np.array([bin_.start for bin_ in firing_rate_bins] + [firing_rate_bins[-1].end])

    bin_vals = np.array([bin_.value for bin_ in firing_rate_bins])
    bin_counts, _ = np.histogram(time_avg_firing_rates, bins=bin_edges)

    return (bin_vals, bin_counts, bin_edges)


def intra_population_average_firing_rate_evolution(populations: Dict[str, PopulationData], spike_recorder: SpikeRecorderData, ts: Iterable[Bin]):
    spike_trains = {name: spike_recorder.get_spike_trains(pop.ids) for name, pop in populations.items()}
    rates = {name: np.array([firing_rates(st, t_bin) for t_bin in ts]) for name, st in spike_trains.items()}
    names = [n for n in populations.keys()]
    avgs = [np.mean(r, axis=1) for r in rates.values()]
    stds = [np.std(r, axis=1) for r in rates.values()]
    return (names, avgs, stds)


def spike_gaps_coefficient_of_variation(population: PopulationData, spike_recorder: SpikeRecorderData, min_nspikes: int = 2) -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(population.ids)
    spike_gaps = (np.diff(st) for st in spike_trains if len(st) >= min_nspikes)
    return np.array([variation(g) for g in spike_gaps])


def spike_gaps_coefficient_of_variation_histogram(population: PopulationData, spike_recorder: SpikeRecorderData, nbins: int, min_nspikes: int = 2) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    cv = spike_gaps_coefficient_of_variation(population, spike_recorder, min_nspikes)
    bin_counts, bin_edges = np.histogram(cv, bins=nbins)
    bin_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
    return (bin_vals, bin_counts, bin_edges)


def firing_rate_temporal_correlation(reference_neuron: NeuronData, neurons: PopulationData, spike_recorder: SpikeRecorderData, t_bins: Iterable[Bin]) -> NDArray[np.float64]:
    fr = np.array([firing_rates(spike_recorder.get_spike_trains(neurons.ids), t_bin) for t_bin in t_bins]).T
    fr_ref = np.squeeze(np.array([firing_rates(spike_recorder.get_spike_trains(reference_neuron.ids), t_bin) for t_bin in t_bins]))
    return np.array([pearsonr(fr_ref, fr[n]).statistic for n in range(len(neurons))])


def pairwise_temporal_correlation_matrix(population: PopulationData, spike_recorder: SpikeRecorderData, t_bins: Sequence[Bin]) -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(population.ids)
    firing_rate_matrix = np.array([firing_rates(spike_trains, t) for t in t_bins])
    return np.corrcoef(firing_rate_matrix, rowvar=False)


def spike_counts_spatial_autocorrelation(neurons: PopulationData, spike_recorder: SpikeRecorderData, xs: Iterable[Interval], ys: Iterable[Interval], ts: Iterable[Bin], max_x_offset: int, max_y_offset: int) -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(neurons.ids)
    counts_map_for_each_t = np.array([[[spike_counts_in_spacetime_region(neurons.x_pos, neurons.y_pos, spike_trains, x_bin, y_bin, t_bin)
                                        for y_bin in ys] for x_bin in xs] for t_bin in ts])
    x_offsets = np.arange(-max_x_offset, max_x_offset + 1)
    y_offsets = np.arange(-max_y_offset, max_y_offset + 1)
    offset_vectors = np.array(tuple(product(x_offsets, y_offsets)), dtype=np.int64)
    correlation = np.array([pearson_correlation_offset_data(m, m, offset_vectors) for m in counts_map_for_each_t])
    correlation_formatted = np.transpose([[correlation[:,i * len(y_offsets) + j] for j in range(len(y_offsets))] for i in range(len(x_offsets))])
    return correlation_formatted


def hayleighs_spatial_autocorrelation(neurons: PopulationData, spike_recorder: SpikeRecorderData, xs: Iterable[Interval], ys: Iterable[Interval], ts: Iterable[Interval]) -> NDArray[np.float64]:
    useless_param1 = None
    useless_param2 = None
    hayleighs_autocorr_wrapper = partial(get_autocorr, rough_patch_size=useless_param1, norm=useless_param2)
    
    spike_trains = spike_recorder.get_spike_trains(neurons.ids)
    counts_map_for_each_t = np.array([[[spike_counts_in_spacetime_region(neurons.x_pos, neurons.y_pos, spike_trains, x_bin, y_bin, t_bin)
                                        for y_bin in ys] for x_bin in xs] for t_bin in ts])
    return np.array([hayleighs_autocorr_wrapper(m) for m in counts_map_for_each_t])


def hayleighs_spatial_autocorrelation_radial_avg(neurons: PopulationData, spike_recorder: SpikeRecorderData, xs: Iterable[Interval], ys: Iterable[Interval], ts: Iterable[Interval], spatial_bin_size: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    microns_per_pixel = spatial_bin_size * 1000
    spatial_autocorr = hayleighs_spatial_autocorrelation(neurons, spike_recorder, xs, ys, ts)
    spectrum, distance = average_over_interp2d(spatial_autocorr, microns_per_pixel/1000, 360)
    return np.array(spectrum, dtype=np.float64), np.array(distance[0,:], dtype=np.float64)


def sigrids_spatial_autocorrelation(neurons: PopulationData, spike_recorder: SpikeRecorderData, xs: Iterable[Interval], ys: Iterable[Interval], ts: Iterable[Interval], overlap_ratio: float = 0.2, mode: Literal['same','full'] = 'same') -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(neurons.ids)
    counts_map_for_each_t = np.array([[[spike_counts_in_spacetime_region(neurons.x_pos, neurons.y_pos, spike_trains, x_bin, y_bin, t_bin)
                                        for y_bin in ys] for x_bin in xs] for t_bin in ts])
    mask = np.ones_like(counts_map_for_each_t[0])
    return np.array([cross_correlate_masked(m, m, mask, mask, mode, axes=(-2,-1), overlap_ratio=overlap_ratio) for m in counts_map_for_each_t])


def sigrids_spatial_autocorrelation_radial_avg(neurons: PopulationData, spike_recorder: SpikeRecorderData, xs: Iterable[Interval], ys: Iterable[Interval], ts: Iterable[Interval]) -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(neurons.ids)
    counts_map_for_each_t = np.array([[[spike_counts_in_spacetime_region(neurons.x_pos, neurons.y_pos, spike_trains, x_bin, y_bin, t_bin)
                                        for y_bin in ys] for x_bin in xs] for t_bin in ts])
    mask = np.ones_like(counts_map_for_each_t[0])
    return get_radial_acorr(counts_map_for_each_t, mask)

