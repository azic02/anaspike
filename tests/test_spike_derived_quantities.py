import unittest
from unittest.mock import MagicMock

import numpy as np

from anaspike.functions.spike_derived_quantities import (spike_counts,
                                                         firing_rates,
                                                         spike_counts_in_spacetime_region)
from anaspike.dataclasses import SpikeTrainArray
from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.histogram import EquiBins



class SharedSetup(unittest.TestCase):
    def setUp(self):
        self.x_pos = np.array((0, 1, 2))
        self.y_pos = np.array((0, 1, 2))
        self.spike_trains = SpikeTrainArray([
                    np.array([10, 20, 30, 40, 50], dtype=np.float64),
                    np.array([15, 25, 35], dtype=np.float64),
                    np.array([], dtype=np.float64 )
                    ])

        self.t_interval_all = MagicMock(
            contains=MagicMock(side_effect=lambda times: (times >= -np.inf) & (times < np.inf)),
            width=np.inf
        )
        self.t_interval_partial = MagicMock(
            contains=MagicMock(side_effect=lambda times: (times >= 15) & (times < 40)),
            width=20
        )
        self.t_interval_empty = MagicMock(
            contains=MagicMock(side_effect=lambda times: (times >= 60) & (times < 70)),
            width=10
        )

        self.x_interval_all = MagicMock(
            contains=MagicMock(side_effect=lambda x: (x >= -np.inf) & (x < np.inf))
        )
        self.x_interval_partial = MagicMock(
            contains=MagicMock(side_effect=lambda x: (x >= 0) & (x < 2))
        )

        self.y_interval_all = MagicMock(
            contains=MagicMock(side_effect=lambda y: (y >= -np.inf) & (y < np.inf))
        )
        self.y_interval_partial = MagicMock(
            contains=MagicMock(side_effect=lambda y: (y >= 1) & (y < 3))
        )


class TestSpikeCounts(SharedSetup):
    def test_all_time(self):
        counts = spike_counts(self.spike_trains, self.t_interval_all)
        np.testing.assert_array_equal(counts, np.array([5, 3, 0]))

    def test_partial_time(self):
        counts = spike_counts(self.spike_trains, self.t_interval_partial)
        np.testing.assert_array_equal(counts, np.array([2, 3, 0]))

    def test_no_spikes(self):
        counts = spike_counts(self.spike_trains, self.t_interval_empty)
        np.testing.assert_array_equal(counts, np.array([0, 0, 0]))

    def test_single_neuron(self):
        spike_train = SpikeTrainArray([self.spike_trains[0]])
        counts = spike_counts(spike_train, self.t_interval_partial)
        np.testing.assert_array_equal(counts, np.array([2]))


class TestFiringRates(SharedSetup):
    def test_all_time(self):
        rates = firing_rates(self.spike_trains, self.t_interval_all)
        expected_rates = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(rates, expected_rates)

    def test_partial_time(self):
        rates = firing_rates(self.spike_trains, self.t_interval_partial)
        expected_rates = np.array([2 / 20 * 1.e3, 3 / 20 * 1.e3, 0])
        np.testing.assert_array_almost_equal(rates, expected_rates)

    def test_no_spikes(self):
        rates = firing_rates(self.spike_trains, self.t_interval_empty)
        expected_rates = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(rates, expected_rates)

    def test_single_neuron(self):
        spike_train = SpikeTrainArray([self.spike_trains[0]])
        rates = firing_rates(spike_train, self.t_interval_partial)
        expected_rates = np.array([2 / self.t_interval_partial.width * 1.e3])
        np.testing.assert_array_almost_equal(rates, expected_rates)


class TestFiringRatesOverTime(SharedSetup):
    def setUp(self):
        super().setUp()
        self.times = EquiBins.from_interval_with_median_values(Interval(0., 50.), 5)

    def test_all_neurons(self):
        rates = np.array([firing_rates(self.spike_trains, t_bin) for t_bin in self.times]).T
        expected_rates = np.array([
            [0, 1.e2, 1.e2, 1.e2, 1.e2],
            [0, 1.e2, 1.e2, 1.e2, 0],
            [0, 0, 0, 0, 0]
        ])
        np.testing.assert_array_almost_equal(rates, expected_rates)

    def test_single_neuron(self):
        spike_train = SpikeTrainArray([self.spike_trains[1]])
        rates = np.squeeze(np.array([firing_rates(spike_train, t_bin) for t_bin in self.times])).T
        expected_rates = np.array([0, 1.e2, 1.e2, 1.e2, 0])
        np.testing.assert_array_almost_equal(rates, expected_rates)


class TestSpikeCountsInSpacetimeRegion(SharedSetup):
    def test_all_spikes_in_intervals(self):
        counts = spike_counts_in_spacetime_region(self.x_pos,
                                                  self.y_pos,
                                                  self.spike_trains,
                                                  self.x_interval_all,
                                                  self.y_interval_all,
                                                  self.t_interval_all)
        expected_counts = 5 + 3
        self.assertEqual(counts, expected_counts)

    def test_some_spikes_in_intervals(self):
        counts = spike_counts_in_spacetime_region(self.x_pos,
                                                  self.y_pos,
                                                  self.spike_trains,
                                                  self.x_interval_partial,
                                                  self.y_interval_partial,
                                                  self.t_interval_partial)
        expected_counts = 3
        self.assertEqual(counts, expected_counts)


if __name__ == '__main__':
    unittest.main(verbosity=2)

