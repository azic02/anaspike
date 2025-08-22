import numpy as np
from numpy.typing import NDArray

from ...hdf5_mixin import HDF5Mixin
from ...dataclasses.nest_devices import PopulationData, SpikeRecorderData
from ...dataclasses.histogram import EquiBins, Histogram



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

    @classmethod
    def from_nest(cls, pop: PopulationData, sr: SpikeRecorderData, time_bins: EquiBins) -> "InstantaneousFiringRates":
        spike_trains = sr.get_spike_trains(pop.ids)
        spike_counts = np.array([Histogram.construct_by_counting(bins=time_bins, data=st).counts for st in spike_trains])
        return cls(
            times=time_bins.values,
            firing_rates=np.array(spike_counts) / time_bins.bin_width * 1.e3
        )

    @property
    def times(self) -> NDArray[np.float64]:
        return self.__times

    @property
    def _firing_rates(self) -> NDArray[np.float64]:
        return self.along_neuron_dim

    @property
    def along_time_dim(self):
        return self.__firing_rates.T

    @property
    def along_neuron_dim(self):
        return self.__firing_rates

    @property
    def n_times(self) -> int:
        return self.times.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.along_neuron_dim.shape[0]

