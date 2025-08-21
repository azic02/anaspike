import unittest

import numpy as np

from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.histogram import EquiBins
from anaspike.analysis.instantaneous_firing_rate import FiringRateEvolution



class TestFiringRateEvolution(unittest.TestCase):
    def setUp(self):
        self.firing_rates = np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])
        self.firing_rate_evolution = FiringRateEvolution(
            time_bins=EquiBins.from_interval_with_median_values(Interval(start=0.0, end=3.0), n=3),
            firing_rates=np.array(self.firing_rates),
        )

    def test_hdf5(self):
        import h5py
        with h5py.File('test_firing_rate_evolution.h5', 'w') as f:
            self.firing_rate_evolution.to_hdf5(f, 'firing_rate_evolution')
        with h5py.File('test_firing_rate_evolution.h5', 'r') as f:
            loaded = FiringRateEvolution.from_hdf5(f['firing_rate_evolution'])
        np.testing.assert_array_equal(loaded.firing_rates, self.firing_rate_evolution.firing_rates)
        np.testing.assert_array_equal(loaded.time_bins.bin_edges, self.firing_rate_evolution.time_bins.bin_edges)

    def test_along_time_dim(self):
        for actual, expected in zip(
            self.firing_rate_evolution.along_time_dim(),
            self.firing_rates
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_along_neuron_dim(self):
        for actual, expected in zip(
            self.firing_rate_evolution.along_neuron_dim(),
            self.firing_rates.T
        ):
            np.testing.assert_array_equal(actual, expected)

