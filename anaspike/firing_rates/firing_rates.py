from typing import Union, TypeVar

import numpy as np
from numpy.typing import NDArray, DTypeLike

from ..hdf5_mixin import HDF5Mixin
from ..spike_counts import SpikeCounts
from ..spike_trains import SpikeTrains
from ..dataclasses.interval import Interval
from ..dataclasses.coords2d import Coords2D
from ..dataclasses.spatial_map import SpatialMap
from ..dataclasses.grid_map_2d import GridMap2D
from ..dataclasses.grid import RectilinearGrid2D



class FiringRates(SpatialMap[Coords2D, np.float64], HDF5Mixin):
    def __init__(self, coords: Coords2D, firing_rates: NDArray[np.float64]):
        if firing_rates.ndim != 1:
            raise ValueError("firing_rates must be a 1D array")
        super().__init__(coords, firing_rates)

    @classmethod
    def from_spike_counts(cls, spike_counts: SpikeCounts, time_window: Interval, time_unit: float=1.e-3):
        return cls(spike_counts.coords, spike_counts.counts / (time_window.width * time_unit))

    @classmethod
    def from_spike_trains(cls, spike_trains: SpikeTrains, time_window: Interval, time_unit: float=1.e-3):
        spike_counts = SpikeCounts.from_spike_trains(spike_trains, time_window)
        return cls.from_spike_counts(spike_counts, time_window, time_unit)

    @property
    def firing_rates(self) -> NDArray[np.float64]:
        return self.values

    def __array__(self, dtype: Union[DTypeLike, None] = None, copy: Union[bool, None] = None):
        return np.array(self.firing_rates, dtype=dtype, copy=copy)


Grid2dT = TypeVar("Grid2dT", bound=RectilinearGrid2D)
BinnedFiringRates = GridMap2D[Grid2dT, np.float64]

