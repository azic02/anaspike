from typing import Union

import numpy as np
from numpy.typing import NDArray, DTypeLike

from ..dataclasses.cartesian_map_2d import CartesianMap2D
from ..dataclasses.coords2d import Coords2D
from ..dataclasses.interval import Interval
from ..spike_trains import SpikeTrains



class SpikeCounts(CartesianMap2D[np.int64]):
    def __init__(self, coords: Coords2D, counts: NDArray[np.int64]):
        if len(coords) != len(counts):
            raise ValueError("Length of `coords` must match length of `counts`.")
        super().__init__(coords, counts)

    @classmethod
    def from_spike_trains(cls, spike_trains: SpikeTrains, time_window: Interval, time_unit: float=1.e-3):
        counts = np.array([np.sum(time_window.contains(st).astype(np.int64)) for st in spike_trains])
        return cls(spike_trains.coords, counts)

    @property
    def counts(self) -> NDArray[np.int64]:
        return self.values

    def __array__(self, dtype: Union[DTypeLike, None] = None, copy: Union[bool, None] = None):
        return np.array(self.counts, dtype=dtype, copy=copy)

    @property
    def n_neurons(self) -> int:
        return len(self.coords)

