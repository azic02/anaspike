import unittest
from unittest.mock import patch, MagicMock

import numpy as np



class TestConstructSpikeTimeHistogram(unittest.TestCase):
    @patch('anaspike.dataclasses.spike_train.SpikeTrainArray')
    @patch('anaspike.dataclasses.histogram.ContigBins')
    def setUp(self, MockSpikeTrainArray: MagicMock, MockContigBins: MagicMock):
        self.mock_spike_trains = MockSpikeTrainArray()
        self.mock_spike_trains.__iter__.return_value = [[0, 0.1, 1.2, 1.4], [1.0, 1.2, 2.4], [0.3, 7.0]]

        self.mock_t_bins = MockContigBins()
        self.mock_t_bins.__len__.return_value = 3
        self.mock_t_bins.bin_edges = [0, 1, 2, 3]

        self.expected_counts = [3, 4, 1]
        self.expected_bin_edges = self.mock_t_bins.bin_edges
        self.expected_bin_values = self.mock_t_bins.bin_values

    def test(self):
        from anaspike.analysis.spike_trains import construct_spike_time_histogram
        result = construct_spike_time_histogram(self.mock_spike_trains, self.mock_t_bins)
        np.testing.assert_array_equal(result.counts, self.expected_counts)
        np.testing.assert_array_almost_equal(result.bin_edges, self.expected_bin_edges)
        np.testing.assert_array_almost_equal(result.bin_values, self.expected_bin_values)


class TestConstructInterspikeIntervalHistogram(unittest.TestCase):
    @patch('anaspike.dataclasses.spike_train.SpikeTrainArray')
    @patch('anaspike.dataclasses.histogram.ContigBins')
    def setUp(self, MockSpikeTrainArray: MagicMock, MockContigBins: MagicMock):
        self.mock_spike_trains = MockSpikeTrainArray()
        self.mock_spike_trains.__iter__.return_value = [[0, 0.1, 1.2, 1.4], [1.0, 1.2, 2.4], [0.3, 7.0]]
        self.mock_spike_trains.__len__.return_value = 3

        self.mock_t_bins = MockContigBins()
        self.mock_t_bins.__len__.return_value = 3
        self.mock_t_bins.bin_edges = [0, 1, 2, 3]

        self.expected_counts = [3, 2, 0]
        self.expected_bin_edges = self.mock_t_bins.bin_edges
        self.expected_bin_values = self.mock_t_bins.bin_values

    def test(self):
        from anaspike.analysis.spike_trains import construct_interspike_interval_histogram
        result = construct_interspike_interval_histogram(self.mock_spike_trains, self.mock_t_bins)
        np.testing.assert_array_equal(result.counts, self.expected_counts)
        np.testing.assert_array_almost_equal(result.bin_edges, self.expected_bin_edges)
        np.testing.assert_array_almost_equal(result.bin_values, self.expected_bin_values)


class CalculateActiveNeuronFractionTestCase(unittest.TestCase):
    @classmethod
    @patch('anaspike.dataclasses.spike_train.SpikeTrainArray')
    @patch('anaspike.dataclasses.interval.Interval')
    def setUpClass(cls, MockSpikeTrainArray: MagicMock, MockInterval: MagicMock):
        cls.MockSpikeTrainArray = MockSpikeTrainArray

        cls.mock_t_interval = MockInterval()
        cls.mock_t_interval.start = 0
        cls.mock_t_interval.end = 2
        cls.mock_t_interval.contains.side_effect = lambda t: (cls.mock_t_interval.start <= t) & (t < cls.mock_t_interval.end)
        cls.thresh = 1

class TestCalculateActiveNeuronFractionSomeActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.mock_spike_trains = self.MockSpikeTrainArray()
        self.mock_spike_trains.n_neurons = 4
        self.mock_spike_trains.__iter__.return_value = [np.array([0, 0.1]),
                                                        np.array([1.0, 1.2]),
                                                        np.array([0.3]),
                                                        np.array([])]

        self.expected_fraction = 3. / 4

    def test(self):
        from anaspike.analysis.spike_trains import calculate_active_neuron_fraction
        result = calculate_active_neuron_fraction(self.mock_spike_trains, self.mock_t_interval, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)

class TestCalculateActiveNeuronFractionNoneActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.mock_spike_trains = self.MockSpikeTrainArray()
        self.mock_spike_trains.n_neurons = 3
        self.mock_spike_trains.__iter__.return_value = [np.array([]),
                                                        np.array([]),
                                                        np.array([])]

        self.expected_fraction = 0.0

    def test(self):
        from anaspike.analysis.spike_trains import calculate_active_neuron_fraction
        result = calculate_active_neuron_fraction(self.mock_spike_trains, self.mock_t_interval, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)

class TestCalculateActiveNeuronFractionAllActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.mock_spike_trains = self.MockSpikeTrainArray()
        self.mock_spike_trains.n_neurons = 3
        self.mock_spike_trains.__iter__.return_value = [np.array([0]),
                                                        np.array([1.0]),
                                                        np.array([0.3])]

        self.expected_fraction = 1.0

    def test(self):
        from anaspike.analysis.spike_trains import calculate_active_neuron_fraction
        result = calculate_active_neuron_fraction(self.mock_spike_trains, self.mock_t_interval, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)


if __name__ == '__main__':
    unittest.main(verbosity=2)

