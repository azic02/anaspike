from typing import Iterable, Sequence, Dict, Tuple, Literal
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.stats import variation

from anaspike.dataclasses.nest_devices import PopulationData, NeuronData, SpikeRecorderData
from anaspike.dataclasses.interval import Interval, Bin, EquidistantBins
from anaspike.functions.spike_derived_quantities import (firing_rates,
                                                         spike_counts_in_spacetime_region)
from anaspike.functions.statistical_quantities import (pearson_correlation_offset_data,
                                                       radial_average,
                                                       morans_i)
from anaspike.hayleigh import get_autocorr, average_over_interp2d
from anaspike.sigrid import cross_correlate_masked, get_radial_acorr
from anaspike.functions._helpers import (construct_offset_vectors,
                                         calculate_pairwise_2d_euclidean_distances)



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


def temporal_correlation_from_firing_rates(ref_firing_rate: NDArray[np.float64], firing_rates: NDArray[np.float64]) -> NDArray[np.float64]:
    if ref_firing_rate.ndim != 1:
        raise ValueError("ref_firing_rate must be a 1D array")
    if firing_rates.ndim != 2:
        raise ValueError("firing_rates must be a 2D array")
    if firing_rates.shape[1] != ref_firing_rate.shape[0]:
        raise ValueError("First dim of firing_rates must match ref_firing_rate")
    return np.array([np.corrcoef(ref_firing_rate, fr)[0, 1] for fr in firing_rates])


def firing_rate_temporal_correlation(reference_neuron: NeuronData, neurons: PopulationData, spike_recorder: SpikeRecorderData, t_bins: Iterable[Bin]) -> NDArray[np.float64]:
    fr = np.array([firing_rates(spike_recorder.get_spike_trains(neurons.ids), t_bin) for t_bin in t_bins]).T
    fr_ref = np.squeeze(np.array([firing_rates(spike_recorder.get_spike_trains(reference_neuron.ids), t_bin) for t_bin in t_bins]))
    return temporal_correlation_from_firing_rates(fr_ref, fr)


def pairwise_temporal_correlation_matrix(population: PopulationData, spike_recorder: SpikeRecorderData, t_bins: Sequence[Bin]) -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(population.ids)
    firing_rate_matrix = np.array([firing_rates(spike_trains, t) for t in t_bins])
    return np.corrcoef(firing_rate_matrix, rowvar=False)


def spike_counts_spatial_autocorrelation(neurons: PopulationData, spike_recorder: SpikeRecorderData, xs: Sequence[Interval], ys: Sequence[Interval], ts: Iterable[Interval]) -> NDArray[np.float64]:
    spike_trains = spike_recorder.get_spike_trains(neurons.ids)
    counts_map_for_each_t = np.array([[[spike_counts_in_spacetime_region(neurons.x_pos, neurons.y_pos, spike_trains, x_bin, y_bin, t_bin)
                                        for y_bin in ys] for x_bin in xs] for t_bin in ts])
    offset_vectors = construct_offset_vectors(len(xs), len(ys), margin=20)
    return np.array([pearson_correlation_offset_data(m, m, offset_vectors) for m in counts_map_for_each_t])


def spike_counts_spatial_autocorrelation_radial_avg(neurons: PopulationData, spike_recorder: SpikeRecorderData, xs: EquidistantBins, ys: EquidistantBins, ts: Iterable[Interval], origin: Tuple[float,float], radial_bins: Sequence[Bin]) -> NDArray[np.float64]:
    spatial_autocorr = spike_counts_spatial_autocorrelation(neurons, spike_recorder, xs, ys, ts)
    offset_vectors = construct_offset_vectors(len(xs), len(ys), margin=20)
    offset_vectors_in_spatial_units = np.array([(origin[0] + xs.bin_width * ov[0], origin[1] + ys.bin_width * ov[1]) for ov in offset_vectors])
    return np.array([radial_average(origin, offset_vectors_in_spatial_units[:,0], offset_vectors_in_spatial_units[:,1], a, radial_bins) for a in spatial_autocorr])


def morans_i_from_firing_rates_evolution(x_pos: NDArray[np.float64], y_pos: NDArray[np.float64], fr_over_t: NDArray[np.float64], decay_power: float=1.) -> NDArray[np.float64]:
    if x_pos.shape != y_pos.shape:
        raise ValueError("x_pos and y_pos must have the same shape")
    if fr_over_t.ndim != 2:
        raise ValueError("fr_over_t must be a 2D array")
    if fr_over_t.shape[1] != x_pos.shape[0]:
        raise ValueError("First dim of fr_over_t must match x_pos")
    distances = calculate_pairwise_2d_euclidean_distances(x_pos, y_pos)
    np.fill_diagonal(distances, 1.)
    weights = np.power(distances, -decay_power)
    np.fill_diagonal(weights, 0.)
    return np.array([morans_i(fr, weights) for fr in fr_over_t])


def morans_i_evolution(neurons: PopulationData, spike_recorder: SpikeRecorderData, ts: Iterable[Bin], decay_power: float=1.) -> NDArray[np.float64]:
    fr_over_t = firing_rates_evolution(neurons, spike_recorder, ts)
    return morans_i_from_firing_rates_evolution(neurons.x_pos, neurons.y_pos, fr_over_t, decay_power)


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

