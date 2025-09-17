import unittest

import numpy as np

from anaspike.analysis.spike_counts import SpikeCounts
from anaspike.dataclasses.spike_train import SpikeTrainArray
from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.coords2d import Coords2D



class TestSpikeCountsFromSpikeTrains(unittest.TestCase):
    def setUp(self):
        self.coords = Coords2D(np.arange(3), np.arange(3))
        self.spike_trains = SpikeTrainArray([
                    np.array([10, 20, 30, 40, 50], dtype=np.float64),
                    np.array([15, 25, 35], dtype=np.float64),
                    np.array([], dtype=np.float64 )
                    ])

        self.t_interval_all = Interval(-np.inf, np.inf)
        self.t_interval_partial = Interval(15, 40)
        self.t_interval_empty = Interval(60, 70)

    def test_all_time(self):
        counts = SpikeCounts.from_spike_trains(self.coords, self.spike_trains, self.t_interval_all)
        np.testing.assert_array_equal(counts, np.array([5, 3, 0]))

    def test_partial_time(self):
        counts = SpikeCounts.from_spike_trains(self.coords, self.spike_trains, self.t_interval_partial)
        np.testing.assert_array_equal(counts, np.array([2, 3, 0]))

    def test_no_spikes(self):
        counts = SpikeCounts.from_spike_trains(self.coords, self.spike_trains, self.t_interval_empty)
        np.testing.assert_array_equal(counts, np.array([0, 0, 0]))

    def test_single_neuron(self):
        spike_train = SpikeTrainArray([self.spike_trains[0]])
        counts = SpikeCounts.from_spike_trains(self.coords[0], spike_train, self.t_interval_partial)
        np.testing.assert_array_equal(counts, np.array([2]))

