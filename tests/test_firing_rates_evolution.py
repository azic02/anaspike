import unittest

import numpy as np

from anaspike.firing_rates_evolution import FiringRatesEvolution



class TestFiringRatesEvolutionClassMethods(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.bins import ContigBins1D
        from anaspike.dataclasses.grid import RegularGrid1D
        from anaspike.dataclasses.interval import Interval

        n_bins = 3

        self.time_bins = ContigBins1D[RegularGrid1D].with_median_labels(
                RegularGrid1D.from_interval_given_n(Interval(0.0, 3.0),
                                                    n=n_bins + 1,
                                                    endpoint=True))

    def test_from_spike_trains(self):
        from anaspike.spike_trains import SpikeTrains
        from anaspike.dataclasses.coords2d import Coords2D
        coords = Coords2D([0, 1, 2], [0, 1, 2])
        spike_trains = SpikeTrains(coords, [
            np.array([0.1, 0.5]),
            np.array([1.1, 2.1, 2.5]),
            np.array([0.75])
        ])
        expected_times = self.time_bins.labels
        expected_firing_rates=np.array([[2000., 0000., 0000.],
                                        [0000., 1000., 2000.],
                                        [1000., 0000., 0000.]]).T

        ifr = FiringRatesEvolution.from_spike_trains(
            spike_trains=spike_trains,
            time_bins=self.time_bins
        )
        np.testing.assert_array_equal(ifr.times, expected_times)
        np.testing.assert_array_equal(ifr.values_time_major, expected_firing_rates)

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

        ifr = FiringRatesEvolution.from_nest(
            pop=population_data,
            sr=spike_recorder_data,
            time_bins=self.time_bins
        )
        np.testing.assert_array_equal(ifr.times, expected_times)
        np.testing.assert_array_equal(ifr.values_time_major, expected_firing_rates)


class TestFiringRatesEvolutionProperties(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.coords2d import Coords2D
        coords = Coords2D([0, 1, 2], [0, 1, 2])
        self.firing_rates = np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])
        self.firing_rates_evolutions = FiringRatesEvolution(
            coords=coords,
            times= np.array([0.0, 1.0, 2.0]),
            values=np.array(self.firing_rates),
        )

    def test_hdf5(self):
        import h5py
        with h5py.File('test_firing_rates_evolutions.h5', 'w') as f:
            self.firing_rates_evolutions.to_hdf5(f, 'firing_rates_evolutions')
        with h5py.File('test_firing_rates_evolutions.h5', 'r') as f:
            loaded = FiringRatesEvolution.from_hdf5(f['firing_rates_evolutions'])
        np.testing.assert_array_equal(loaded.values_neurons_major, self.firing_rates_evolutions.values_neurons_major)
        np.testing.assert_array_equal(loaded.times, self.firing_rates_evolutions.times)

    def test_values_time_major(self):
        for actual, expected in zip(
            self.firing_rates_evolutions.values_time_major,
            self.firing_rates.T
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_values_neurons_major(self):
        for actual, expected in zip(
            self.firing_rates_evolutions.values_neurons_major,
            self.firing_rates
        ):
            np.testing.assert_array_equal(actual, expected)

