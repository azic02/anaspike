from typing import Union

import numpy as np
from numpy.typing import NDArray, DTypeLike

from ...hdf5_mixin import HDF5Mixin
from ...dataclasses import SpikeTrainArray
from ...dataclasses.interval import Interval



class TimeAveragedFiringRate(HDF5Mixin):
    def __init__(self, firing_rates: NDArray[np.float64]):
        if firing_rates.ndim != 1:
            raise ValueError("firing_rates must be a 1D array")

        self.__firing_rates = np.array(firing_rates, dtype=np.float64)

    @classmethod
    def from_spike_trains(cls, spike_trains: SpikeTrainArray, time_window: Interval, time_unit: float=1.e-3):
        spike_counts = np.array([np.sum(time_window.contains(st)) for st in spike_trains])
        return cls(spike_counts / (time_window.width * time_unit))

    @property
    def as_nparray(self) -> NDArray[np.float64]:
        return self.__firing_rates

    def __array__(self, dtype: Union[DTypeLike, None] = None, copy: Union[bool, None] = None):
        return np.array(self.__firing_rates, dtype=dtype, copy=copy)

