import unittest

import numpy as np

from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.bins import ContigBins1D
from anaspike.dataclasses.spike_train import SpikeTrainArray
from anaspike.dataclasses.grid import Grid1D



class TestConstructSpikeTimeHistogram(unittest.TestCase):
    def setUp(self):
        self.mock_spike_trains = SpikeTrainArray([np.array([0, 0.1, 1.2, 1.4]),
                                                  np.array([1.0, 1.2, 2.4]),
                                                  np.array([0.3, 7.0])])

        self.mock_t_bins = ContigBins1D[Grid1D].with_median_values(Grid1D(np.array([0, 1, 2, 3])))

        self.expected_counts = [3, 4, 1]
        self.expected_bin_edges = self.mock_t_bins.edges
        self.expected_bin_values = self.mock_t_bins.values

    def test(self):
        from anaspike.analysis.spike_trains import construct_spike_time_histogram
        result = construct_spike_time_histogram(self.mock_spike_trains, self.mock_t_bins)
        np.testing.assert_array_equal(result.counts, self.expected_counts)
        np.testing.assert_array_almost_equal(result.edges, self.expected_bin_edges)
        np.testing.assert_array_almost_equal(result.values, self.expected_bin_values)


class TestConstructInterspikeIntervalHistogram(unittest.TestCase):
    def setUp(self):
        self.mock_spike_trains = SpikeTrainArray([np.array([0, 0.1, 1.2, 1.4]),
                                                  np.array([1.0, 1.2, 2.4]),
                                                  np.array([0.3, 7.0])])

        self.mock_t_bins = ContigBins1D[Grid1D].with_median_values(Grid1D(np.array([0, 1, 2, 3])))

        self.expected_counts = [3, 2, 0]
        self.expected_bin_edges = self.mock_t_bins.edges
        self.expected_bin_values = self.mock_t_bins.values

    def test(self):
        from anaspike.analysis.spike_trains import construct_interspike_interval_histogram
        result = construct_interspike_interval_histogram(self.mock_spike_trains, self.mock_t_bins)
        np.testing.assert_array_equal(result.counts, self.expected_counts)
        np.testing.assert_array_almost_equal(result.edges, self.expected_bin_edges)
        np.testing.assert_array_almost_equal(result.values, self.expected_bin_values)


class CalculateActiveNeuronFractionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_t_interval = Interval(0, 2)
        cls.thresh = 1

class TestCalculateActiveNeuronFractionSomeActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.mock_spike_trains = SpikeTrainArray([np.array([0, 0.1]),
                                                  np.array([1.0, 1.2]),
                                                  np.array([0.3]),
                                                  np.array([])])

        self.expected_fraction = 3. / 4

    def test(self):
        from anaspike.analysis.spike_trains import calculate_active_neuron_fraction
        result = calculate_active_neuron_fraction(self.mock_spike_trains, self.mock_t_interval, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)

class TestCalculateActiveNeuronFractionNoneActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.mock_spike_trains = SpikeTrainArray([np.array([]),
                                                  np.array([]),
                                                  np.array([])])

        self.expected_fraction = 0.0

    def test(self):
        from anaspike.analysis.spike_trains import calculate_active_neuron_fraction
        result = calculate_active_neuron_fraction(self.mock_spike_trains, self.mock_t_interval, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)

class TestCalculateActiveNeuronFractionAllActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.mock_spike_trains = SpikeTrainArray([np.array([0]),
                                                  np.array([1.0]),
                                                  np.array([0.3])])

        self.expected_fraction = 1.0

    def test(self):
        from anaspike.analysis.spike_trains import calculate_active_neuron_fraction
        result = calculate_active_neuron_fraction(self.mock_spike_trains, self.mock_t_interval, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)


if __name__ == '__main__':
    unittest.main(verbosity=2)

