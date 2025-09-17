import unittest

import numpy as np

from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.histogram import Histogram
from anaspike.dataclasses.bins import ContigBins1D
from anaspike.dataclasses.grid import Grid1D, RegularGrid1D


class TestHistogramInit(unittest.TestCase):
    def test_non_matching_lengths(self):
        bins = ContigBins1D(edges=Grid1D(np.array([0, 2, 4])), values=np.array([1, 3]))
        counts = np.array([10, 20, 30])
        with self.assertRaises(ValueError):
            Histogram(bins=bins, counts=counts)

    def test_valid_init(self):
        edges = Grid1D(np.array([0, 2, 4, 6, 8, 10]))
        values = np.array([1, 3, 5, 7, 9])
        bins = ContigBins1D(edges=edges, values=values)
        counts = np.array([10, 20, 30, 40, 50])
        histogram = Histogram(bins=bins, counts=counts)

        np.testing.assert_array_equal(histogram.bins.edges, edges)
        np.testing.assert_array_equal(histogram.bins.values, values)
        np.testing.assert_array_equal(histogram.counts, counts)


class TestHistogramConstructByCounting(unittest.TestCase):
    def test_construct_by_counting(self):
        bins = ContigBins1D[Grid1D].with_median_values(edges=Grid1D(np.array([0, 2, 4, 6, 8, 10])))
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 3.6, 3.7, 5.6])
        histogram = Histogram.construct_by_counting(bins, data)

        expected_counts = np.array([1, 4, 3, 2, 1])
        np.testing.assert_array_equal(histogram.counts, expected_counts)

    def test_construct_by_counting_regular_grid(self):
        bins = ContigBins1D[RegularGrid1D].with_median_values(RegularGrid1D.given_n_with_endpoint(Interval(0, 10), 6))
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 3.6, 3.7, 5.6])
        histogram = Histogram.construct_by_counting(bins, data)

        expected_counts = np.array([1, 4, 3, 2, 1])
        np.testing.assert_array_equal(histogram.counts, expected_counts)


class TestHistogram(unittest.TestCase):
    def setUp(self):
        edges = Grid1D(np.array([0, 2, 4, 6, 8, 10]))
        values = np.array([1, 3, 5, 7, 9])
        bins = ContigBins1D(edges=edges, values=values)
        counts = np.array([10, 20, 30, 40, 50])
        self.histogram = Histogram(bins=bins, counts=counts)

    def test_bins(self):
        expected_edges = np.array([0, 2, 4, 6, 8, 10])
        expected_values = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.histogram.bins.edges, expected_edges)
        np.testing.assert_array_equal(self.histogram.bins.values, expected_values)

    def test_edges(self):
        expected_edges = np.array([0, 2, 4, 6, 8, 10])
        np.testing.assert_array_equal(self.histogram.bins.edges, expected_edges)

    def test_values(self):
        expected_values = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.histogram.bins.values, expected_values)

    def test_counts(self):
        expected_counts = np.array([10, 20, 30, 40, 50])
        np.testing.assert_array_equal(self.histogram.counts, expected_counts)

    def test_hdf5_conversion(self):
        import h5py
        with h5py.File('test_histogram.h5', 'w') as f_out:
            self.histogram.to_hdf5(f_out, 'histogram')
        with h5py.File('test_histogram.h5', 'r') as f_in:
            loaded_histogram = Histogram.from_hdf5(f_in['histogram'])
        np.testing.assert_array_equal(loaded_histogram.bins.edges, self.histogram.bins.edges)
        np.testing.assert_array_equal(loaded_histogram.bins.values, self.histogram.bins.values)
        np.testing.assert_array_equal(loaded_histogram.counts, self.histogram.counts)


if __name__ == '__main__':
    unittest.main(verbosity=2)

