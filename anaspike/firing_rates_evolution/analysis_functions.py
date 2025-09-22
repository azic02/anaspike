from typing import cast

import numpy as np
from numpy.typing import NDArray

from .firing_rates_evolution import FiringRatesEvolution
from anaspike.dataclasses.coords2d import Coords2D
from anaspike.functions._helpers import calculate_pairwise_2d_euclidean_distances
from anaspike.functions.statistical_quantities import morans_i



def temporal_correlation(firing_rates: FiringRatesEvolution,
                         ref_firing_rate: NDArray[np.float64]) -> NDArray[np.float64]:
    if ref_firing_rate.ndim != 1:
        raise ValueError("Reference firing rate must be a 1D array.")
    if ref_firing_rate.shape[0] != firing_rates.n_times:
        raise ValueError("Reference and comparison firing rates must have the same number of time points.")
    return np.array([cast(float, np.corrcoef(ref_firing_rate, fr)[0, 1]) for fr in firing_rates.along_neuron_dim])


def temporal_correlation_matrix(firing_rates: FiringRatesEvolution) -> NDArray[np.float64]:
    return np.corrcoef(firing_rates.along_neuron_dim).astype(np.float64)


def morans_i_evolution(firing_rates: FiringRatesEvolution, coords: Coords2D, decay_power: float=1.) -> NDArray[np.float64]:
    if firing_rates.n_neurons != len(coords):
        raise ValueError("Number of neurons in `firing_rates` and length of `coords` must match.")
    distances = calculate_pairwise_2d_euclidean_distances(coords.x, coords.y)
    np.fill_diagonal(distances, 1.)
    weights = np.power(distances, -decay_power)
    np.fill_diagonal(weights, 0.)
    return np.array([morans_i(fr, weights) for fr in firing_rates.along_time_dim])

