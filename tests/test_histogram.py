import unittest

import numpy as np

from anaspike.dataclasses.interval import Interval, Bin
from anaspike.dataclasses.histogram import (ContigBins,
                                            EquiBins,
                                            Histogram)

class TestContigBinsInit(unittest.TestCase):
    def test_invalid_edges_dim(self):
        with self.assertRaises(ValueError):
            ContigBins(edges=np.array([[0, 1], [2, 3]]), values=np.array([1, 2]))

    def test_invalid_values_dim(self):
        with self.assertRaises(ValueError):
            ContigBins(edges=np.array([0, 1, 2]), values=np.array([[1], [2]]))

    def test_too_few_edges(self):
        with self.assertRaises(ValueError):
            ContigBins(edges=np.array([0]), values=np.array([]))

    def test_too_few_values(self):
        with self.assertRaises(ValueError):
            ContigBins(edges=np.array([0, 1, 2]), values=np.array([1]))

    def test_too_many_values(self):
        with self.assertRaises(ValueError):
            ContigBins(edges=np.array([0, 1, 2]), values=np.array([1, 2, 3]))

    def test_non_monotonic_increasing_edges(self):
        with self.assertRaises(ValueError):
            ContigBins(edges=np.array([0, 2, 1]), values=np.array([1, 2]))

    def test_valid_init(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        values = np.array([1, 3, 5, 7, 9])
        bins = ContigBins(edges=edges, values=values)

        np.testing.assert_array_equal(bins.bin_edges, edges)
        np.testing.assert_array_equal(bins.bin_values, values)


class TestContigBinsFromBinSequence(unittest.TestCase):
    def setUp(self):
        self.bins = [Bin(start=0, end=2, value=1),
                     Bin(start=2, end=4, value=3),
                     Bin(start=4, end=6, value=5),
                     Bin(start=6, end=8, value=7),
                     Bin(start=8, end=10, value=9)]

    def test_negative_max_gap(self):
        with self.assertRaises(ValueError):
            ContigBins.from_bin_sequence(self.bins, max_gap=-1.e-8)

    def test_negative_max_overlap(self):
        with self.assertRaises(ValueError):
            ContigBins.from_bin_sequence(self.bins, max_overlap=-1.e-8)

    def test_max_gap_exceeded(self):
        bins = self.bins + [Bin(start=10.00001, end=12, value=11)]
        with self.assertRaises(ValueError):
            ContigBins.from_bin_sequence(bins, max_gap=1.e-8)

    def test_max_overlap_exceeded(self):
        bins = self.bins + [Bin(start=0.99999, end=12, value=11)]
        with self.assertRaises(ValueError):
            ContigBins.from_bin_sequence(bins, max_overlap=1.e-8)

    def test_valid_bins(self):
        contig_bins = ContigBins.from_bin_sequence(self.bins, max_gap=1.e-8, max_overlap=1.e-8)
        expected_edges = np.array([0, 2, 4, 6, 8, 10])
        expected_values = np.array([1, 3, 5, 7, 9])

        np.testing.assert_array_equal(contig_bins.bin_edges, expected_edges)
        np.testing.assert_array_equal(contig_bins.bin_values, expected_values)


class TestContigBinsWithMedianValues(unittest.TestCase):
    def test_valid(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        bins = ContigBins.with_median_values(edges)

        expected_values = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(bins.bin_edges, edges)
        np.testing.assert_array_equal(bins.bin_values, expected_values)


class TestContigBins(unittest.TestCase):
    def setUp(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        values = np.array([1, 3, 5, 7, 9])
        self.bins = ContigBins(edges, values)

    def test_bin_edges(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        np.testing.assert_array_equal(self.bins.bin_edges, edges)

    def test_bin_values(self):
        values = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.bins.bin_values, values)

    def test_len(self):
        self.assertEqual(len(self.bins), 5)

    def test_getitem(self):
        expected_bins = [Bin(start=0, end=2, value=1),
                         Bin(start=2, end=4, value=3),
                         Bin(start=4, end=6, value=5),
                         Bin(start=6, end=8, value=7),
                         Bin(start=8, end=10, value=9)]

        for i in range(5):
            bin_ = self.bins[i]
            self.assertEqual(bin_.start, expected_bins[i].start)
            self.assertEqual(bin_.end, expected_bins[i].end)
            self.assertEqual(bin_.value, expected_bins[i].value)

        for i in range(-5, 0):
            bin_ = self.bins[i]
            self.assertEqual(bin_.start, expected_bins[i].start)
            self.assertEqual(bin_.end, expected_bins[i].end)
            self.assertEqual(bin_.value, expected_bins[i].value)

        with self.assertRaises(IndexError):
            last_valid_idx = len(self.bins) - 1
            _ = self.bins[last_valid_idx + 1]

        with self.assertRaises(IndexError):
            last_valid_reverse_idx = -len(self.bins)
            _ = self.bins[last_valid_reverse_idx - 1]

    def test_iter(self):
        expected_bins = [Bin(start=0, end=2, value=1),
                         Bin(start=2, end=4, value=3),
                         Bin(start=4, end=6, value=5),
                         Bin(start=6, end=8, value=7),
                         Bin(start=8, end=10, value=9)]

        bins_iter = iter(self.bins)
        for i, bin_ in enumerate(bins_iter):
            self.assertEqual(bin_.start, expected_bins[i].start)
            self.assertEqual(bin_.end, expected_bins[i].end)
            self.assertEqual(bin_.value, expected_bins[i].value)

        with self.assertRaises(StopIteration):
            next(bins_iter)

    def test_hdf5_conversion(self):
        import h5py

        with h5py.File('test_bins.h5', 'w') as f:
            out_group = f.create_group('bins')
            self.bins.to_hdf5(out_group)

        with h5py.File('test_bins.h5', 'r') as f:
            in_group = f['bins']
            loaded_bins = ContigBins.from_hdf5(in_group)

        np.testing.assert_array_equal(loaded_bins.bin_edges, self.bins.bin_edges)
        np.testing.assert_array_equal(loaded_bins.bin_values, self.bins.bin_values)


class TestEquiBinsInit(unittest.TestCase):
    def test_invalid_width(self):
        edges = np.array([0, 2, 4.00001, 6, 8, 10])
        values = np.array([1, 3, 5, 7, 9])
        with self.assertRaises(ValueError):
            EquiBins(edges=edges, values=values, rtol=1.e-6, atol=1.e-8)

    def test_valid_init(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        values = np.array([1, 3, 5, 7, 9])
        bins = EquiBins(edges=edges, values=values)

        np.testing.assert_array_equal(bins.bin_edges, edges)
        np.testing.assert_array_equal(bins.bin_values, values)


class TestEquiBinsFromIntervalWithMedianValues(unittest.TestCase):
    def setUp(self):
        self.interval = Interval(0, 10)

    def test_given_n(self):
        result = EquiBins.from_interval_with_median_values(self.interval, n=5)
        expected_edges = np.array([0, 2, 4, 6, 8, 10])
        expected_values = np.array([1, 3, 5, 7, 9])

        np.testing.assert_array_equal(result.bin_edges, expected_edges)
        np.testing.assert_array_equal(result.bin_values, expected_values)

    def test_given_size(self):
        result = EquiBins.from_interval_with_median_values(self.interval, size=2)
        expected_edges = np.array([0, 2, 4, 6, 8])
        expected_values = np.array([1, 3, 5, 7])

        np.testing.assert_array_equal(result.bin_edges, expected_edges)
        np.testing.assert_array_equal(result.bin_values, expected_values)


    def test_given_size_float(self):
        result = EquiBins.from_interval_with_median_values(Interval(100.,10000.), size=500.)
        expected_edges = np.array([100., 600., 1100., 1600., 2100., 2600.,
                                   3100., 3600., 4100., 4600., 5100., 5600.,
                                   6100., 6600., 7100., 7600., 8100., 8600.,
                                   9100., 9600.])
        expected_values = np.array([350., 850., 1350., 1850., 2350., 2850.,
                                    3350., 3850., 4350., 4850., 5350., 5850.,
                                    6350., 6850., 7350., 7850., 8350., 8850.,
                                    9350.])

        np.testing.assert_array_equal(result.bin_edges, expected_edges)
        np.testing.assert_array_equal(result.bin_values, expected_values)


class TestEquiBins(unittest.TestCase):
    def setUp(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        values = np.array([1, 3, 5, 7, 9])
        self.bins = EquiBins(edges=edges, values=values)

    def test_bin_width(self):
        expected_width = 2.0
        self.assertAlmostEqual(self.bins.bin_width, expected_width)


class TestHistogramInit(unittest.TestCase):
    def test_non_matching_lengths(self):
        bins = ContigBins(edges=np.array([0, 2, 4]), values=np.array([1, 3]))
        counts = np.array([10, 20, 30])
        with self.assertRaises(ValueError):
            Histogram(bins=bins, counts=counts)

    def test_valid_init(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        values = np.array([1, 3, 5, 7, 9])
        bins = ContigBins(edges=edges, values=values)
        counts = np.array([10, 20, 30, 40, 50])
        histogram = Histogram(bins=bins, counts=counts)

        np.testing.assert_array_equal(histogram.bins.bin_edges, edges)
        np.testing.assert_array_equal(histogram.bins.bin_values, values)
        np.testing.assert_array_equal(histogram.counts, counts)


class TestHistogramConstructByCounting(unittest.TestCase):
    def test_construct_by_counting(self):
        bins = ContigBins.with_median_values(edges=np.array([0, 2, 4, 6, 8, 10]))
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 3.6, 3.7, 5.6])
        histogram = Histogram.construct_by_counting(bins, data)

        expected_counts = np.array([1, 4, 3, 2, 1])
        np.testing.assert_array_equal(histogram.counts, expected_counts)


class TestHistogram(unittest.TestCase):
    def setUp(self):
        edges = np.array([0, 2, 4, 6, 8, 10])
        values = np.array([1, 3, 5, 7, 9])
        bins = ContigBins(edges=edges, values=values)
        counts = np.array([10, 20, 30, 40, 50])
        self.histogram = Histogram(bins=bins, counts=counts)

    def test_bins(self):
        expected_edges = np.array([0, 2, 4, 6, 8, 10])
        expected_values = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.histogram.bins.bin_edges, expected_edges)
        np.testing.assert_array_equal(self.histogram.bins.bin_values, expected_values)

    def test_bin_edges(self):
        expected_edges = np.array([0, 2, 4, 6, 8, 10])
        np.testing.assert_array_equal(self.histogram.bins.bin_edges, expected_edges)

    def test_bin_values(self):
        expected_values = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.histogram.bins.bin_values, expected_values)

    def test_counts(self):
        expected_counts = np.array([10, 20, 30, 40, 50])
        np.testing.assert_array_equal(self.histogram.counts, expected_counts)

    def test_hdf5_conversion(self):
        import h5py

        with h5py.File('test_histogram.h5', 'w') as f:
            out_group = f.create_group('histogram')
            self.histogram.to_hdf5(out_group)

        with h5py.File('test_histogram.h5', 'r') as f:
            in_group = f['histogram']
            loaded_histogram = Histogram.from_hdf5(in_group)

        np.testing.assert_array_equal(loaded_histogram.bins.bin_edges, self.histogram.bins.bin_edges)
        np.testing.assert_array_equal(loaded_histogram.bins.bin_values, self.histogram.bins.bin_values)
        np.testing.assert_array_equal(loaded_histogram.counts, self.histogram.counts)


if __name__ == '__main__':
    unittest.main(verbosity=2)

