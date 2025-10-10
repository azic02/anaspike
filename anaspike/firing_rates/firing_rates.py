from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from ..hdf5_mixin import HDF5Mixin
from ..spike_counts import SpikeCounts
from ..spike_trains import SpikeTrains
from ..dataclasses.interval import Interval
from ..dataclasses.coords2d import Coords2D
from ..dataclasses.scalar_spatial_map import ScalarSpatialMap
from ..dataclasses.scalar_spatial_grid_map import ScalarSpatialGridMap
from ..dataclasses.grid import RectilinearGrid2D



class FiringRates(ScalarSpatialMap[Coords2D], HDF5Mixin):
    def __init__(self, coords: Coords2D, values: NDArray[np.float64]):
        super().__init__(coords, values)

    @classmethod
    def from_spike_counts(cls, spike_counts: SpikeCounts, time_window: Interval, time_unit: float=1.e-3):
        return cls(spike_counts.coords, spike_counts.counts / (time_window.width * time_unit))

    @classmethod
    def from_spike_trains(cls, spike_trains: SpikeTrains, time_window: Interval, time_unit: float=1.e-3):
        spike_counts = SpikeCounts.from_spike_trains(spike_trains, time_window)
        return cls.from_spike_counts(spike_counts, time_window, time_unit)


Grid2dT = TypeVar("Grid2dT", bound=RectilinearGrid2D)
BinnedFiringRates = ScalarSpatialGridMap[Grid2dT]

