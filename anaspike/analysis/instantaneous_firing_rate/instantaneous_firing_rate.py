import numpy as np
from numpy.typing import NDArray

from ...hdf5_mixin import HDF5Mixin
from ...dataclasses.histogram import ContigBins



class FiringRateEvolution(HDF5Mixin):
    def __init__(self, time_bins: ContigBins, firing_rates: NDArray[np.float64]):
        if firing_rates.ndim != 2:
            raise ValueError("`firing_rates` must be a 2D array.")
        if len(time_bins) != firing_rates.shape[0]:
            raise ValueError("First dimension of `firing_rates` must match length of `time_bins`.")

        self.__time_bins = time_bins
        self.__firing_rates = firing_rates

    @property
    def time_bins(self) -> ContigBins:
        return self.__time_bins

    @property
    def firing_rates(self) -> NDArray[np.float64]:
        return self.__firing_rates

    def along_time_dim(self):
        return self.__firing_rates

    def along_neuron_dim(self):
        return self.__firing_rates.T

