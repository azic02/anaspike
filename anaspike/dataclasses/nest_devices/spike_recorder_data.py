
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ...functions._helpers import validate_same_length



@dataclass
class SpikeRecorderData:
    senders: NDArray[np.int64]
    times: NDArray[np.float64]

    def __post_init__(self):
        validate_same_length(self.senders, self.times)

    @classmethod
    def from_nest_ascii_backend(cls, filepaths: Iterable[Path]) -> 'SpikeRecorderData':
        return cls(senders=np.hstack([np.loadtxt(p, delimiter='\t', skiprows=3, usecols=(0,), dtype=np.int64) for p in filepaths]),
                   times=np.hstack([np.loadtxt(p, delimiter='\t', skiprows=3, usecols=(1,), dtype=np.float64) for p in filepaths]))

    @classmethod
    def from_hdf5(cls, entity) -> 'SpikeRecorderData':
        return cls(senders=entity['senders'][()], times=entity['times'][()])

    def to_hdf5(self, entity) -> None:
        entity.create_dataset('senders', data=self.senders)
        entity.create_dataset('times', data=self.times)

    def __add__(self, other: 'SpikeRecorderData') -> 'SpikeRecorderData':
        return SpikeRecorderData(senders=np.hstack([self.senders, other.senders]),
                                  times=np.hstack([self.times, other.times]))

