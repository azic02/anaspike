import unittest

import numpy as np

from anaspike.spike_trains import SpikeTrains
from anaspike.dataclasses.bins import ContigBins1D
from anaspike.dataclasses.grid import Grid1D
from anaspike.dataclasses.coords2d import Coords2D



class TestConstructSpikeTimeHistogram(unittest.TestCase):
    def setUp(self):
        coords = Coords2D([0, 1, 2], [0, 1, 2])
        self.spike_trains = SpikeTrains(coords, [np.array([0, 0.1, 1.2, 1.4]),
                                                 np.array([1.0, 1.2, 2.4]),
                                                 np.array([0.3, 7.0])])

        self.t_bins = ContigBins1D[Grid1D].with_median_labels(Grid1D(np.array([0, 1, 2, 3])))

        self.expected_counts = [3, 4, 1]
        self.expected_bin_edges = self.t_bins.edges
        self.expected_bin_labels = self.t_bins.labels

    def test(self):
        from anaspike.spike_trains import construct_spike_time_histogram
        result = construct_spike_time_histogram(self.spike_trains, self.t_bins)
        np.testing.assert_array_equal(result.counts, self.expected_counts)
        np.testing.assert_array_almost_equal(result.edges, self.expected_bin_edges)
        np.testing.assert_array_almost_equal(result.labels, self.expected_bin_labels)


class TestConstructInterspikeIntervalHistogram(unittest.TestCase):
    def setUp(self):
        coords = Coords2D([0, 1, 2], [0, 1, 2])
        self.spike_trains = SpikeTrains(coords, [np.array([0, 0.1, 1.2, 1.4]),
                                                 np.array([1.0, 1.2, 2.4]),
                                                 np.array([0.3, 7.0])])

        self.t_bins = ContigBins1D[Grid1D].with_median_labels(Grid1D(np.array([0, 1, 2, 3])))

        self.expected_counts = [3, 2, 0]
        self.expected_bin_edges = self.t_bins.edges
        self.expected_bin_labels = self.t_bins.labels

    def test(self):
        from anaspike.spike_trains import construct_interspike_interval_histogram
        result = construct_interspike_interval_histogram(self.spike_trains, self.t_bins)
        np.testing.assert_array_equal(result.counts, self.expected_counts)
        np.testing.assert_array_almost_equal(result.edges, self.expected_bin_edges)
        np.testing.assert_array_almost_equal(result.labels, self.expected_bin_labels)


if __name__ == '__main__':
    unittest.main(verbosity=2)

