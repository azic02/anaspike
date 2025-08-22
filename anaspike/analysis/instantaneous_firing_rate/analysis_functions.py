from typing import cast

import numpy as np
from numpy.typing import NDArray

from .instantaneous_firing_rate import InstantaneousFiringRates



def temporal_correlation(firing_rates: InstantaneousFiringRates,
                         ref_firing_rate: NDArray[np.float64]) -> NDArray[np.float64]:
    if ref_firing_rate.ndim != 1:
        raise ValueError("Reference firing rate must be a 1D array.")
    if ref_firing_rate.shape[0] != firing_rates.n_times:
        raise ValueError("Reference and comparison firing rates must have the same number of time points.")
    return np.array([cast(float, np.corrcoef(ref_firing_rate, fr)[0, 1]) for fr in firing_rates.along_neuron_dim])
