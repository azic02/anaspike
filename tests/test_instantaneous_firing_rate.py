import unittest

import numpy as np

from anaspike.analysis.instantaneous_firing_rate import InstantaneousFiringRates



class TestInstantaneousFiringRatesInit(unittest.TestCase):
    def test_valid_input(self):
        times = np.array([0.0, 1.0, 2.0])
        firing_rates = np.array([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]])
        ifr = InstantaneousFiringRates(times=times, firing_rates=firing_rates)
        self.assertTrue(np.array_equal(ifr.times, times))
        self.assertTrue(np.array_equal(ifr.along_neuron_dim, firing_rates))

    def test_invalid_times_shape(self):
        with self.assertRaises(ValueError):
            InstantaneousFiringRates(times=np.array([[0.0], [1.0]]), firing_rates=np.array([[1.0]]))

    def test_invalid_firing_rates_shape(self):
        with self.assertRaises(ValueError):
            InstantaneousFiringRates(times=np.array([0.0]), firing_rates=np.array([1.0]))

    def test_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            InstantaneousFiringRates(times=np.array([0.0, 1.0]),
                                     firing_rates=np.array([[1.0],
                                                            [2.0]]))


class TestInstantaneousFiringRatesClassMethods(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.bins import ContigBins1D
        from anaspike.dataclasses.grid import RegularGrid1D
        from anaspike.dataclasses.interval import Interval

        n_bins = 3

        self.time_bins = ContigBins1D[RegularGrid1D].with_median_labels(RegularGrid1D.given_n_with_endpoint(Interval(0.0, 3.0), n=n_bins + 1))

    def test_from_spike_trains(self):
        from anaspike.dataclasses.spike_train import SpikeTrainArray
        spike_trains = SpikeTrainArray([
            np.array([0.1, 0.5]),
            np.array([1.1, 2.1, 2.5]),
            np.array([0.75])
        ])
        expected_times = self.time_bins.labels
        expected_firing_rates=np.array([[2000., 0000., 0000.],
                                        [0000., 1000., 2000.],
                                        [1000., 0000., 0000.]]).T

        ifr = InstantaneousFiringRates.from_spike_trains(
            spike_trains=spike_trains,
            time_bins=self.time_bins
        )
        np.testing.assert_array_equal(ifr.times, expected_times)
        np.testing.assert_array_equal(ifr.along_time_dim, expected_firing_rates)

    def test_from_nest(self):
        from anaspike.dataclasses.nest_devices import PopulationData, SpikeRecorderData
        population_data = PopulationData(
            ids=np.array([0, 1, 2]),
            x_pos=np.array([0.1, 2.0, 0.25]),
            y_pos=np.array([0.5, 2.5, 0.75])
        )

        spike_recorder_data = SpikeRecorderData(
                senders=np.array([0, 0, 2, 1, 1, 1], dtype=np.int64),
                times=np.array([0.1, 0.5, 0.75, 1.1, 2.1, 2.5], dtype=np.float64)
                )

        expected_times = self.time_bins.labels
        expected_firing_rates=np.array([[2000., 0000., 0000.],
                                        [0000., 1000., 2000.],
                                        [1000., 0000., 0000.]]).T

        ifr = InstantaneousFiringRates.from_nest(
            pop=population_data,
            sr=spike_recorder_data,
            time_bins=self.time_bins
        )
        np.testing.assert_array_equal(ifr.times, expected_times)
        np.testing.assert_array_equal(ifr.along_time_dim, expected_firing_rates)


class TestInstantaneousFiringRatesProperties(unittest.TestCase):
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
        np.testing.assert_array_equal(loaded.along_neuron_dim, self.instantaneous_firing_rates.along_neuron_dim)
        np.testing.assert_array_equal(loaded.times, self.instantaneous_firing_rates.times)

    def test_along_time_dim(self):
        for actual, expected in zip(
            self.instantaneous_firing_rates.along_time_dim,
            self.firing_rates.T
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_along_neuron_dim(self):
        for actual, expected in zip(
            self.instantaneous_firing_rates.along_neuron_dim,
            self.firing_rates
        ):
            np.testing.assert_array_equal(actual, expected)

