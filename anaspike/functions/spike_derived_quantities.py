import numpy as np
from numpy.typing import NDArray

from ..dataclasses import SpikeTrainArray
from ..dataclasses.interval import Interval



def spike_counts(spike_trains: SpikeTrainArray, time_window: Interval=Interval(-np.inf,np.inf)) -> NDArray[np.int64]:
    return np.array([np.sum(time_window.contains(st)) for st in spike_trains])


def firing_rates(spike_trains: SpikeTrainArray, time_window: Interval) -> NDArray[np.float64]:
    return spike_counts(spike_trains, time_window) / time_window.width * 1.e3


def spike_counts_in_spacetime_region(x_pos: NDArray[np.float64],
                                     y_pos: NDArray[np.float64],
                                     spike_trains: SpikeTrainArray,
                                     x_interval: Interval,
                                     y_interval: Interval,
                                     t_interval: Interval
                                     ) -> np.int64:
    mask = x_interval.contains(x_pos) & y_interval.contains(y_pos)
    return np.sum(spike_counts(spike_trains[mask], t_interval))

