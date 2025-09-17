from typing import Union

import numpy as np
from numpy.typing import NDArray, DTypeLike

from ...hdf5_mixin import HDF5Mixin
from ..spike_trains import SpikeTrains
from ...dataclasses.interval import Interval
from ...dataclasses.coords2d import Coords2D
from ...dataclasses.field import Field2D



class TimeAveragedFiringRate(Field2D[np.float64], HDF5Mixin):
    def __init__(self, coords: Coords2D, firing_rates: NDArray[np.float64]):
        if firing_rates.ndim != 1:
            raise ValueError("firing_rates must be a 1D array")
        super().__init__(coords, firing_rates)

    @classmethod
    def from_spike_trains(cls, coords: Coords2D, spike_trains: SpikeTrains, time_window: Interval, time_unit: float=1.e-3):
        spike_counts = np.array([np.sum(time_window.contains(st)) for st in spike_trains])
        return cls(coords, spike_counts / (time_window.width * time_unit))

    @property
    def firing_rates(self) -> NDArray[np.float64]:
        return self.elements

    def __array__(self, dtype: Union[DTypeLike, None] = None, copy: Union[bool, None] = None):
        return np.array(self.firing_rates, dtype=dtype, copy=copy)

    @property
    def shape(self):
        return self.firing_rates.shape



