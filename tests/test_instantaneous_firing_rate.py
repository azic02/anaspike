import unittest

import numpy as np

from anaspike.analysis.instantaneous_firing_rate import InstantaneousFiringRates



class TestInstantaneousFiringRates(unittest.TestCase):
    def setUp(self):
        self.firing_rates = np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])
        self.instantaneous_firing_rates = InstantaneousFiringRates(
            times= np.array([0.0, 1.0, 2.0]),
            firing_rates=np.array(self.firing_rates),
        )

    def test_hdf5(self):
        import h5py
        with h5py.File('test_instantaneous_firing_rates.h5', 'w') as f:
            self.instantaneous_firing_rates.to_hdf5(f, 'instantaneous_firing_rates')
        with h5py.File('test_instantaneous_firing_rates.h5', 'r') as f:
            loaded = InstantaneousFiringRates.from_hdf5(f['instantaneous_firing_rates'])
        np.testing.assert_array_equal(loaded.firing_rates, self.instantaneous_firing_rates.firing_rates)
        np.testing.assert_array_equal(loaded.times, self.instantaneous_firing_rates.times)

    def test_along_time_dim(self):
        for actual, expected in zip(
            self.instantaneous_firing_rates.along_time_dim,
            self.firing_rates
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_along_neuron_dim(self):
        for actual, expected in zip(
            self.instantaneous_firing_rates.along_neuron_dim,
            self.firing_rates.T
        ):
            np.testing.assert_array_equal(actual, expected)

