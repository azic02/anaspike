import numpy as np
from numpy.typing import NDArray
from typing import Collection



SpikeTrain = NDArray[np.float64]

class SpikeTrainArray(np.ndarray):
    def __new__(cls, spike_trains: Collection[SpikeTrain]):
        arr = np.asarray(spike_trains, dtype=object)
        return arr.view(cls)

