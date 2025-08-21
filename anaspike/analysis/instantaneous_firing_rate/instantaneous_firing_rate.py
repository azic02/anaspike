import numpy as np
from numpy.typing import NDArray

from ...hdf5_mixin import HDF5Mixin



class InstantaneousFiringRates(HDF5Mixin):
    def __init__(self, times: NDArray[np.float64], firing_rates: NDArray[np.float64]):
        if times.ndim != 1:
            raise ValueError("`times` must be a 1D array.")
        if firing_rates.ndim != 2:
            raise ValueError("`firing_rates` must be a 2D array.")
        if times.shape[0] != firing_rates.shape[0]:
            raise ValueError("First dimensions of `firing_rates` and `times` must match.")

        self.__times = times
        self.__firing_rates = firing_rates

    @property
    def times(self) -> NDArray[np.float64]:
        return self.__times

    @property
    def firing_rates(self) -> NDArray[np.float64]:
        return self.__firing_rates

    @property
    def along_time_dim(self):
        return self.firing_rates

    @property
    def along_neuron_dim(self):
        return self.firing_rates.T

    @property
    def n_times(self) -> int:
        return self.times.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.firing_rates.shape[1]

